const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000";

export async function sendChat(
  question,
  sessionId
) {
  const response = await fetch(
    `${API_BASE_URL}/api/chat`,
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    }
  );

  if (!response.ok) {
    let message =
      "법률 상담 요청에 실패했습니다.";

    try {
      const errorData =
        await response.json();

      if (errorData.detail) {
        message =
          typeof errorData.detail ===
            "string"
            ? errorData.detail
            : JSON.stringify(
              errorData.detail
            );
      }
    } catch {
      // JSON 에러가 아니라면 기본 메시지 사용
    }

    throw new Error(message);
  }

  return response.json();
}