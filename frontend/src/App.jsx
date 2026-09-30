import { useState } from "react";
import { useConversations } from "./hooks/useConversations";
import "./App.css";
import Header from "./components/layout/Header";
import Sidebar from "./components/layout/sidebar";
import ChatInput from "./components/ChatInput";
import ChatWindow from "./components/chat/Chatwindow";
import { sendChat } from "./api/chatApi";

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

function App() {
  const { conversations, setConversations, sortedConversations } =
    useConversations();

  const [activeChatId, setActiveChatId] = useState(null);
  const [sessionId, setSessionId] = useState(createSessionId);
  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

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

        <ChatWindow messages={messages} loading={loading} onSend={handleSend} />

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
