export async function sendChat(question, sessionId = "web-user") {
    const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            question,
            session_id: sessionId,
        }),
    });

    if (!response.ok) {
        throw new Error("법률 상담 요청에 실패했습니다.");
    }

    return response.json();
}