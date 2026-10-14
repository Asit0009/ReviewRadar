import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { MonitoringResponse } from '@/api/types'
import { COLORS } from '@/lib/theme'

/** PSI above 0.1 is a moderate shift, above 0.2 a large one (thresholds from reviewpulse/monitoring/drift_checker.py). */
export const PSI_WATCH = 0.1
export const PSI_ALERT = 0.2

export function DriftChart({ features, height = 150 }: { features: MonitoringResponse['feature_metrics']; height?: number }) {
  const rows = Object.entries(features).map(([name, f]) => ({ name: name.replace(/_/g, ' '), psi: f.psi, drifted: f.drift_detected }))
  const xMax = Math.max(0.25, ...rows.map((r) => r.psi * 1.15))
  return (
    <div style={{ height }} role="img" aria-label={`Population stability index per feature: ${rows.map((r) => `${r.name} ${r.psi}`).join(', ')}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 36, bottom: 0, left: 0 }} barSize={14}>
          <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 4" horizontal={false} />
          <XAxis type="number" domain={[0, xMax]} tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={{ stroke: COLORS.grid }} tickCount={6} />
          <YAxis type="category" dataKey="name" width={86} tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={false} />
          <ReferenceLine x={PSI_WATCH} stroke={COLORS.neutral} strokeDasharray="4 3" />
          <ReferenceLine x={PSI_ALERT} stroke={COLORS.negative} strokeDasharray="4 3" />
          <Bar dataKey="psi" radius={[0, 6, 6, 0]} isAnimationActive={false}>
            {rows.map((r) => (
              <Cell key={r.name} fill={r.drifted ? COLORS.negative : COLORS.positive} />
            ))}
            <LabelList dataKey="psi" position="right" fill={COLORS.axis} fontSize={11} formatter={(v: unknown) => (typeof v === 'number' ? v.toFixed(3) : '')} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
