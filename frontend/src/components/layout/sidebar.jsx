function Sidebar({
  conversations,
  activeChatId,
  sidebarOpen,
  loading,
  onClose,
  onNewChat,
  onSelect,
  onDelete,
}) {
  return (
    <>
      {sidebarOpen && (
        <button
          type="button"
          className="sidebar-overlay"
          onClick={onClose}
          aria-label="상담 기록 닫기"
        />
      )}

      <aside
        className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}
        aria-label="상담 기록"
      >
        <div className="sidebar-header">
          <div>
            <h2>상담 기록</h2>
            <p>이전 상담을 다시 확인하세요.</p>
          </div>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={onNewChat}
          disabled={loading}
        >
          + 새 상담
        </button>

        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div className="conversation-empty">
              아직 저장된 상담이 없습니다.
            </div>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`conversation-row ${
                  activeChatId === conversation.id ? "active" : ""
                }`}
              >
                <button
                  type="button"
                  className={`conversation-item ${
                    activeChatId === conversation.id ? "active" : ""
                  }`}
                  onClick={() => onSelect(conversation)}
                  disabled={loading}
                >
                  <div className="conversation-info">
                    <span className="conversation-title">
                      {conversation.title}
                    </span>

                    <span className="conversation-date">
                      {new Date(conversation.updatedAt).toLocaleString(
                        "ko-KR",
                        {
                          month: "numeric",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        },
                      )}
                    </span>
                  </div>
                </button>

                <button
                  type="button"
                  className="conversation-delete"
                  aria-label={`${conversation.title} 상담 삭제`}
                  onClick={(event) => onDelete(event, conversation.id)}
                  disabled={loading}
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
