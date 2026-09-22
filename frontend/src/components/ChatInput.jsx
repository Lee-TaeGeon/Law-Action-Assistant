import { useState } from "react";

function ChatInput({ onSend, loading }) {
  const [value, setValue] = useState("");

  const submitQuestion = () => {
    const question = value.trim();

    if (!question || loading) {
      return;
    }

    onSend(question);
    setValue("");
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    submitQuestion();
  };

  const handleKeyDown = (e) => {
    if (
      e.key === "Enter" &&
      !e.shiftKey
    ) {
      e.preventDefault();
      submitQuestion();
    }
  };

  return (
    <div className="chat-input-wrapper">
      <form
        className="chat-input-container"
        onSubmit={handleSubmit}
      >
        <textarea
          className="chat-input"
          value={value}
          onChange={(e) =>
            setValue(e.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="법률 상황을 입력해주세요."
          rows={2}
          disabled={loading}
        />

        <button
          className="send-button"
          type="submit"
          disabled={
            loading || !value.trim()
          }
        >
          {loading
            ? "분석 중..."
            : "질문하기"}
        </button>
      </form>
    </div>
  );
}

export default ChatInput;