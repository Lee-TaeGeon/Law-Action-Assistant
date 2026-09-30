function Header({ onOpenSidebar, onNewChat, loading }) {
  return (
    <header className="app-header">
      <div className="header-left">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onOpenSidebar}
          aria-label="상담 기록 열기"
        >
          ☰
        </button>

        <div>
          <h1>Law Action Assistant</h1>

          <p>AI 기반 법률 정보 검색 및 상담 도우미</p>
        </div>
      </div>

      <button
        type="button"
        className="header-new-chat-button"
        onClick={onNewChat}
        disabled={loading}
      >
        + 새 상담
      </button>
    </header>
  );
}

export default Header;
