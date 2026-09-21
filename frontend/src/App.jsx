import { useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import { sendChat } from "./api/chatApi";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSend = async (question) => {
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
      const data = await sendChat(question);

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
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Law Action Assistant</h1>
        <p>법령 근거 기반 AI 법률 상담</p>
      </header>

      <main className="chat-container">
        {messages.length === 0 && (
          <div className="empty-state">
            법률 상황을 입력하면 관련 법령을 검색해 답변합니다.
          </div>
        )}

        {messages.map((message, index) => (
          <ChatMessage
            key={index}
            message={message}
          />
        ))}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
      </main>

      <ChatInput
        onSend={handleSend}
        loading={loading}
      />
    </div>
  );
}

export default App;