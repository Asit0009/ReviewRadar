import { ShoppingBag } from 'lucide-react'
import { useCategories } from '@/api/queries'
import heroBanner from '@/assets/hero-banner.webp'

export function Hero() {
  const { data } = useCategories()
  const categoryCount = data ? `${data.total_categories} product categories` : 'several product categories'
  return (
    <section aria-label="About the dataset" className="relative isolate overflow-hidden rounded-xl border border-line bg-card/90">
      <img
        src={heroBanner}
        alt=""
        width={1200}
        height={670}
        className="pointer-events-none absolute inset-y-0 right-0 -z-10 h-full w-3/5 object-cover object-[80%_50%] opacity-40 [mask-image:linear-gradient(to_right,transparent,black_45%)] sm:w-1/2 sm:opacity-100"
      />
      <div className="flex items-center gap-4 px-5 py-5 sm:px-6 sm:py-6">
        <span aria-hidden="true" className="hidden size-14 shrink-0 items-center justify-center rounded-xl bg-white/95 text-[#0b1230] shadow-lg xs:flex">
          <ShoppingBag size={28} />
        </span>
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-[28px]">Amazon Reviews 2023</h1>
          <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-ink-2">
            Real customer feedback from {categoryCount}, explored with data science and NLP.
          </p>
        </div>
      </div>
    </section>
  )
}
