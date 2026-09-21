import { useState } from "react";

function ChatInput({ onSend, loading }) {
  const [value, setValue] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    const question = value.trim();

    if (!question || loading) {
      return;
    }

    onSend(question);
    setValue("");
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="법률 상황을 입력해주세요."
        rows={3}
      />

      <button type="submit" disabled={loading}>
        {loading ? "분석 중..." : "질문하기"}
      </button>
    </form>
  );
}

export default ChatInput;