import { useEffect, useRef, useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import { sendChat } from "./api/chatApi";
import "./App.css";

function createSessionId() {
  if (crypto?.randomUUID) {
    return crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(() =>
    createSessionId()
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading, error]);

  const handleSend = async (question) => {
    if (!question.trim() || loading) {
      return;
    }

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: question,
      },
    ]);

    setLoading(true);
    setError("");

    try {
      const data = await sendChat(
        question,
        sessionId
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          category: data.category,
          sources: data.sources,
        },
      ]);
    } catch (error) {
      console.error(error);

      setError(
        error.message ||
          "답변을 불러오는 중 문제가 발생했습니다."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    if (loading) {
      return;
    }

    setMessages([]);
    setError("");
    setSessionId(
      createSessionId()
    );
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div>
            <h1>Law Action Assistant</h1>
            <p>
              법령 근거 기반 AI 법률 상담
            </p>
          </div>

          <button
            type="button"
            className="new-chat-button"
            onClick={handleNewChat}
            disabled={loading}
          >
            <span aria-hidden="true">＋</span>
            새 상담
          </button>
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-icon">
              ⚖
            </div>

            <h2>
              어떤 법률 상황이 궁금하신가요?
            </h2>

            <p>
              상황을 입력하면 관련 법령을
              검색하여 근거와 함께
              답변합니다.
            </p>

            <div className="example-questions">
              <span>
                중고거래 사기를 당했어요
              </span>

              <span>
                회사에서 임금을 못 받았어요
              </span>

              <span>
                이혼 시 재산분할이 궁금해요
              </span>
            </div>
          </div>
        )}

        {messages.map(
          (message, index) => (
            <ChatMessage
              key={`${message.role}-${index}`}
              message={message}
            />
          )
        )}

        {loading && (
          <div className="assistant-message loading-message">
            <div className="loading-header">
              <span className="loading-dot" />
              <span>
                관련 법령을 분석하고 있습니다
              </span>
            </div>

            <div className="loading-bars">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        {error && (
          <div
            className="error-message"
            role="alert"
          >
            <strong>
              답변을 불러오지 못했습니다.
            </strong>

            <span>
              {error}
            </span>
          </div>
        )}

        <div ref={chatEndRef} />
      </main>

      <ChatInput
        onSend={handleSend}
        loading={loading}
      />

      <footer className="app-footer">
        AI가 제공하는 내용은 법률정보
        참고용이며 개별 사건에 대한
        법률 자문을 대신하지 않습니다.
      </footer>
    </div>
  );
}

export default App;