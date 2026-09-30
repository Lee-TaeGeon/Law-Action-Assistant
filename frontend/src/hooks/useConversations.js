import { useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "law-action-assistant-chats-v1";

// 저장된 상담 기록 불러오기
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

export function useConversations() {
  const [conversations, setConversations] = useState(
    loadSavedConversations
  );

  // 상담 기록 저장
  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(conversations)
      );
    } catch (error) {
      console.error("상담 기록 저장 실패:", error);
    }
  }, [conversations]);

  // 최근 상담 순서대로 정렬
  const sortedConversations = useMemo(() => {
    return [...conversations].sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() -
        new Date(a.updatedAt).getTime()
    );
  }, [conversations]);

  return {
    conversations,
    setConversations,
    sortedConversations,
  };
}