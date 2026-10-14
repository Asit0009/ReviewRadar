import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

export function AppShell() {
  const { pathname } = useLocation()

  // A new page should start at the top. Filter changes inside a page keep their scroll position.
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only z-[60] rounded-md bg-brand px-3 py-2 text-sm text-white focus:not-sr-only focus:fixed focus:top-2 focus:left-2">
        Skip to content
      </a>
      <Sidebar />
      <div className="lg:pl-[222px]">
        <TopBar />
        <main id="main" className="px-4 pt-4 pb-10 sm:px-5">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
