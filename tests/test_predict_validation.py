"""Strict request validation for POST /predict (and the other request bodies).

The bug these tests guard against: a body such as {"text": "...", "model": "linear_svc"} used to be accepted. The
unknown field `model` was ignored and the default model answered, so the caller believed one model had run when
another had. Now unknown fields, unknown model names and missing fields are all 422 errors, and a valid request
runs exactly the model it names.
"""

from unittest import mock

import pytest

from reviewpulse.modeling.inference import SentimentPredictors

URL = "/api/v1/predict"
TEXT = "The battery drains quickly and it stopped working"
MODELS = {
    "logistic_regression": "TF-IDF + LogisticRegression",
    "linear_svc": "TF-IDF + LinearSVC",
}


@pytest.fixture
def predictors(client):
    return client.app.state.predictors


@pytest.fixture
def must_not_predict():
    """Any request that reaches the predictors fails the test: invalid input must be stopped before a model runs."""
    with mock.patch.object(SentimentPredictors, "predict", side_effect=AssertionError("a model was invoked for an invalid request")) as spy:
        yield spy


def errors(resp):
    assert resp.status_code == 422, resp.text
    return [(e["type"], tuple(e["loc"])) for e in resp.json()["detail"]]


# ---------- unknown fields are errors, never ignored ----------
def test_model_instead_of_model_type_is_rejected_not_defaulted(client, must_not_predict):
    """The original bug: `model` was ignored and logistic regression answered."""
    resp = client.post(URL, json={"text": TEXT, "model": "linear_svc"})
    found = errors(resp)
    assert ("extra_forbidden", ("body", "model")) in found
    assert ("missing", ("body", "model_type")) in found  # and the model choice is required
    must_not_predict.assert_not_called()


def test_an_extra_field_next_to_a_valid_request_is_rejected(client, must_not_predict):
    resp = client.post(URL, json={"text": TEXT, "model_type": "linear_svc", "model": "logistic_regression"})
    assert errors(resp) == [("extra_forbidden", ("body", "model"))]
    must_not_predict.assert_not_called()


@pytest.mark.parametrize(
    "wrong_name",
    ["model", "modelType", "Model_Type", "MODEL_TYPE", "model_name", "model_type ", " model_type", "modeltype", "type", "algorithm", "classifier", "model-type"],
)
def test_wrong_property_names_are_rejected(client, must_not_predict, wrong_name):
    found = errors(client.post(URL, json={"text": TEXT, wrong_name: "linear_svc"}))
    assert ("extra_forbidden", ("body", wrong_name)) in found
    must_not_predict.assert_not_called()


def test_wrong_name_for_text_is_rejected(client, must_not_predict):
    found = errors(client.post(URL, json={"Text": TEXT, "review": TEXT, "model_type": "linear_svc"}))
    assert ("missing", ("body", "text")) in found and ("extra_forbidden", ("body", "Text")) in found
    must_not_predict.assert_not_called()


def test_unrelated_extra_fields_are_rejected(client, must_not_predict):
    assert ("extra_forbidden", ("body", "debug")) in errors(client.post(URL, json={"text": TEXT, "model_type": "linear_svc", "debug": True}))


# ---------- the model choice is explicit ----------
def test_model_type_is_required(client, must_not_predict):
    assert errors(client.post(URL, json={"text": TEXT})) == [("missing", ("body", "model_type"))]


@pytest.mark.parametrize(
    "bad",
    ["", " ", "gpt-4", "random_forest", "LogisticRegression", "LOGISTIC_REGRESSION", "logistic_regression ", " linear_svc", "svc", "bert", "distilbert-base-uncased", "logistic_regression'; --", None, 0, 1, True, ["linear_svc"], {"name": "linear_svc"}],
)
def test_unknown_or_malformed_model_names_are_rejected(client, must_not_predict, bad):
    found = errors(client.post(URL, json={"text": TEXT, "model_type": bad}))
    assert found and all(loc == ("body", "model_type") for _, loc in found)
    must_not_predict.assert_not_called()


def test_the_error_names_the_supported_models(client):
    detail = client.post(URL, json={"text": TEXT, "model_type": "gpt-4"}).json()["detail"][0]
    for name in ("logistic_regression", "linear_svc", "distilbert"):
        assert name in detail["msg"]


# ---------- the text is validated ----------
@pytest.mark.parametrize("bad", [None, "", " ", "   \n\t ", "a", " a ", "x" * 5001, 5, 5.5, True, ["a b"], {"a": "b"}])
def test_invalid_text_is_rejected(client, must_not_predict, bad):
    found = errors(client.post(URL, json={"text": bad, "model_type": "linear_svc"}))
    assert found and all(loc == ("body", "text") for _, loc in found)
    must_not_predict.assert_not_called()


def test_missing_text_is_rejected(client, must_not_predict):
    assert errors(client.post(URL, json={"model_type": "linear_svc"})) == [("missing", ("body", "text"))]


def test_empty_body_reports_both_missing_fields(client, must_not_predict):
    assert set(errors(client.post(URL, json={}))) == {("missing", ("body", "text")), ("missing", ("body", "model_type"))}


@pytest.mark.parametrize("raw", ["[]", "null", '"just a string"', "42", "{not json", ""])
def test_body_that_is_not_a_json_object_is_rejected(client, must_not_predict, raw):
    resp = client.post(URL, content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_text_of_exactly_the_limits_is_accepted(client):
    assert client.post(URL, json={"text": "ok", "model_type": "linear_svc"}).status_code == 200
    assert client.post(URL, json={"text": "a" * 5000, "model_type": "linear_svc"}).status_code == 200


def test_surrounding_spaces_are_removed_before_prediction(client):
    data = client.post(URL, json={"text": "   great product   ", "model_type": "linear_svc"}).json()
    assert data["text"] == "great product"


# ---------- a valid request runs exactly the model it names ----------
@pytest.mark.parametrize("requested", list(MODELS))
def test_the_requested_classical_model_is_the_one_that_runs(client, predictors, requested):
    chosen = predictors.lr_model if requested == "logistic_regression" else predictors.svc_model
    other = predictors.svc_model if requested == "logistic_regression" else predictors.lr_model
    with mock.patch.object(chosen, "predict", wraps=chosen.predict) as chosen_spy, mock.patch.object(other, "predict", wraps=other.predict) as other_spy:
        resp = client.post(URL, json={"text": TEXT, "model_type": requested})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == requested and data["model_used"] == MODELS[requested]
    assert chosen_spy.call_count == 1, "the requested model was not invoked"
    assert other_spy.call_count == 0, "a different model was invoked"
    assert data["predicted_sentiment"] in {"positive", "neutral", "negative"}
    assert data["sentiment_encoded"] == ["negative", "neutral", "positive"].index(data["predicted_sentiment"])


def test_the_two_classical_models_report_different_names_for_the_same_text(client):
    names = {m: client.post(URL, json={"text": TEXT, "model_type": m}).json()["model_used"] for m in MODELS}
    assert len(set(names.values())) == 2


def test_distilbert_request_goes_to_the_distilbert_code_and_to_no_other_model(client, predictors):
    stub = {
        "model_type": "distilbert",
        "predicted_sentiment": "negative",
        "sentiment_encoded": 0,
        "probabilities": {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
        "model_used": "DistilBERT (stub)",
    }
    with mock.patch.object(SentimentPredictors, "_predict_distilbert", return_value=stub) as distil, mock.patch.object(
        predictors.lr_model, "predict", wraps=predictors.lr_model.predict
    ) as lr_spy, mock.patch.object(predictors.svc_model, "predict", wraps=predictors.svc_model.predict) as svc_spy:
        resp = client.post(URL, json={"text": TEXT, "model_type": "distilbert"})
    assert resp.status_code == 200 and resp.json()["model_type"] == "distilbert"
    distil.assert_called_once_with(TEXT)
    assert lr_spy.call_count == 0 and svc_spy.call_count == 0


def test_an_unavailable_model_is_an_error_not_a_different_model(client, predictors):
    """DistilBERT has no saved files in the test data: the answer must be 503, not a logistic-regression result."""
    with mock.patch.object(predictors.lr_model, "predict", wraps=predictors.lr_model.predict) as lr_spy:
        resp = client.post(URL, json={"text": TEXT, "model_type": "distilbert"})
    assert resp.status_code == 503 and "DistilBERT" in resp.json()["detail"]
    assert lr_spy.call_count == 0


def test_predictor_names_the_model_that_ran(predictors):
    for requested, label in MODELS.items():
        result = predictors.predict(TEXT, requested)
        assert result["model_type"] == requested and result["model_used"] == label
    with pytest.raises(ValueError):
        predictors.predict(TEXT, "random_forest")


# ---------- the same strictness for the assistant request ----------
def test_assistant_request_rejects_unknown_fields(client):
    resp = client.post("/api/v1/assistant/chat", json={"question": "battery problems", "category": None, "top_k": 3, "model": "x", "topk": 3})
    found = errors(resp)
    assert ("extra_forbidden", ("body", "model")) in found and ("extra_forbidden", ("body", "topk")) in found


def test_assistant_request_still_accepts_its_documented_fields(client):
    resp = client.post("/api/v1/assistant/chat", json={"question": "battery problems", "category": "Electronics", "top_k": 3})
    assert resp.status_code == 200


# ---------- the OpenAPI contract says the same thing ----------
def test_openapi_documents_the_strict_request(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    predict = schemas["PredictRequest"]
    assert predict["additionalProperties"] is False
    assert set(predict["required"]) == {"text", "model_type"}
    assert predict["properties"]["model_type"]["enum"] == ["logistic_regression", "linear_svc", "distilbert"]
    assert schemas["AssistantChatRequest"]["additionalProperties"] is False
    assert "model_type" in schemas["PredictResponse"]["required"]
