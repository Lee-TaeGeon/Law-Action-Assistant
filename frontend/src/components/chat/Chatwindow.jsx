import ChatMessage from "../ChatMessage";

function ChatWindow({ messages, loading, onSend }) {
  const exampleQuestions = [
    {
      title: "임금을 받지 못했어요",
      question: "회사에서 임금을 두 달째 받지 못했습니다. 어떻게 해야 하나요?",
    },
    {
      title: "중고거래 사기를 당했어요",
      question: "중고거래 사기를 당한 것 같습니다. 어떻게 대응해야 하나요?",
    },
    {
      title: "이혼 재산분할이 궁금해요",
      question: "이혼할 때 재산분할은 어떻게 하나요?",
    },
  ];

  return (
    <section className="chat-container">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">⚖️</div>

          <h2>어떤 법률 문제가 궁금하신가요?</h2>

          <p>상황을 구체적으로 입력하면 관련 법령을 검색해 답변합니다.</p>

          <div className="example-questions">
            {exampleQuestions.map((item) => (
              <button
                key={item.title}
                type="button"
                onClick={() => onSend(item.question)}
                disabled={loading}
              >
                {item.title}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="message-list">
          {messages.map((message, index) => (
            <ChatMessage key={`${message.role}-${index}`} message={message} />
          ))}

          {loading && (
            <div className="loading-message">
              <div className="loading-avatar">AI</div>

              <div className="loading-content">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default ChatWindow;
