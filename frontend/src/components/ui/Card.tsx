import { useId, type ReactNode } from 'react'

interface CardProps {
  title?: ReactNode
  subtitle?: ReactNode
  /** Controls on the right side of the header (tabs, selects, links). */
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}

export function Card({ title, subtitle, actions, children, className = '', bodyClassName = '' }: CardProps) {
  const headingId = useId()
  return (
    <section
      aria-labelledby={title ? headingId : undefined}
      className={`flex min-w-0 flex-col rounded-xl border border-line bg-card/90 shadow-[0_1px_0_rgb(255_255_255/0.03)_inset,0_8px_24px_-12px_rgb(0_0_0/0.6)] ${className}`}
    >
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 px-4 pt-3.5 pb-2">
          <div className="min-w-0">
            {title && (
              <h2 id={headingId} className="text-[15px] leading-tight font-semibold text-ink">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex min-w-0 flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={`min-h-0 flex-1 px-4 pb-4 ${title || actions ? '' : 'pt-4'} ${bodyClassName}`}>{children}</div>
    </section>
  )
}
