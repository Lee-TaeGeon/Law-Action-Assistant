import { useEffect, useMemo, useState } from "react";
import "./App.css";
import Header from "./components/layout/Header";
import Sidebar from "./components/layout/sidebar";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import { sendChat } from "./api/chatApi";

const STORAGE_KEY = "law-action-assistant-chats-v1";

function createSessionId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createTitle(question) {
  const trimmed = question.trim();

  if (trimmed.length <= 28) {
    return trimmed;
  }

  return `${trimmed.slice(0, 28)}...`;
}

function loadSavedConversations() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (!saved) {
      return [];
    }

    const parsed = JSON.parse(saved);

    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("상담 기록 불러오기 실패:", error);
    return [];
  }
}

function App() {
  const [conversations, setConversations] = useState(loadSavedConversations);

  const [activeChatId, setActiveChatId] = useState(null);
  const [sessionId, setSessionId] = useState(createSessionId);
  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (error) {
      console.error("상담 기록 저장 실패:", error);
    }
  }, [conversations]);

  const sortedConversations = useMemo(() => {
    return [...conversations].sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
    );
  }, [conversations]);

  const handleNewChat = () => {
    if (loading) {
      return;
    }

    setActiveChatId(null);
    setSessionId(createSessionId());
    setMessages([]);
    setError("");
    setSidebarOpen(false);
  };

  const handleSelectConversation = (conversation) => {
    if (loading) {
      return;
    }

    setActiveChatId(conversation.id);
    setSessionId(conversation.sessionId);
    setMessages(conversation.messages || []);
    setError("");
    setSidebarOpen(false);
  };

  const handleDeleteConversation = (event, conversationId) => {
    event.stopPropagation();

    if (loading) {
      return;
    }

    const shouldDelete = window.confirm("이 상담 기록을 삭제할까요?");

    if (!shouldDelete) {
      return;
    }

    setConversations((prev) =>
      prev.filter((conversation) => conversation.id !== conversationId),
    );

    if (activeChatId === conversationId) {
      setActiveChatId(null);
      setSessionId(createSessionId());
      setMessages([]);
      setError("");
    }
  };

  const handleSend = async (question) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) {
      return;
    }

    setLoading(true);
    setError("");

    const userMessage = {
      role: "user",
      content: trimmedQuestion,
    };

    const currentChatId = activeChatId || sessionId;

    const currentSessionId = sessionId;

    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);

    setConversations((prev) => {
      const existing = prev.find(
        (conversation) => conversation.id === currentChatId,
      );

      if (existing) {
        return prev.map((conversation) =>
          conversation.id === currentChatId
            ? {
                ...conversation,
                messages: nextMessages,
                updatedAt: new Date().toISOString(),
              }
            : conversation,
        );
      }

      return [
        {
          id: currentChatId,
          sessionId: currentSessionId,
          title: createTitle(trimmedQuestion),
          messages: nextMessages,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
        ...prev,
      ];
    });

    if (!activeChatId) {
      setActiveChatId(currentChatId);
    }

    try {
      const data = await sendChat(trimmedQuestion, currentSessionId);

      const assistantMessage = {
        role: "assistant",
        content: data.answer,
        category: data.category,
        sources: data.sources || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);

      setConversations((prev) =>
        prev.map((conversation) => {
          if (conversation.id !== currentChatId) {
            return conversation;
          }

          return {
            ...conversation,
            messages: [...conversation.messages, assistantMessage],
            updatedAt: new Date().toISOString(),
          };
        }),
      );
    } catch (error) {
      console.error(error);

      setError(error.message || "상담 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        conversations={sortedConversations}
        activeChatId={activeChatId}
        sidebarOpen={sidebarOpen}
        loading={loading}
        onClose={() => setSidebarOpen(false)}
        onNewChat={handleNewChat}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
      />

      <main className="main-panel">
        <Header
          onOpenSidebar={() => setSidebarOpen(true)}
          onNewChat={handleNewChat}
          loading={loading}
        />

        <section className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">⚖️</div>

              <h2>어떤 법률 문제가 궁금하신가요?</h2>

              <p>상황을 구체적으로 입력하면 관련 법령을 검색해 답변합니다.</p>

              <div className="example-questions">
                <button
                  onClick={() =>
                    handleSend(
                      "회사에서 임금을 두 달째 받지 못했습니다. 어떻게 해야 하나요?",
                    )
                  }
                  disabled={loading}
                >
                  임금을 받지 못했어요
                </button>

                <button
                  onClick={() =>
                    handleSend(
                      "중고거래 사기를 당한 것 같습니다. 어떻게 대응해야 하나요?",
                    )
                  }
                  disabled={loading}
                >
                  중고거래 사기를 당했어요
                </button>

                <button
                  onClick={() =>
                    handleSend("이혼할 때 재산분할은 어떻게 하나요?")
                  }
                  disabled={loading}
                >
                  이혼 재산분할이 궁금해요
                </button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <ChatMessage
                  key={`${message.role}-${index}`}
                  message={message}
                />
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

        {error && <div className="error-message">{error}</div>}

        <ChatInput onSend={handleSend} loading={loading} />

        <footer className="app-footer">
          본 서비스는 법률 정보 제공을 위한 보조 도구이며, 전문적인 법률 자문을
          대체하지 않습니다.
        </footer>
      </main>
    </div>
  );
}

export default App;
