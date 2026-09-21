import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SourceCard from "./SourceCard";

function normalizeMarkdown(text = "") {
  return text
    .replace(/\\([#*_`~])/g, "$1")
    .replace(/<br\s*\/?>/gi, "\n");
}

function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="message user-message">
        {message.content}
      </div>
    );
  }

  return (
    <div className="message assistant-message">
      <span className="category-badge">
        {message.category}
      </span>

      <div className="answer markdown-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {normalizeMarkdown(message.content)}
        </ReactMarkdown>
      </div>

      {message.sources?.length > 0 && (
        <div className="sources">
          <h3>관련 법령</h3>

          {message.sources.map((source, index) => (
            <SourceCard
              key={`${source.law_name}-${index}`}
              source={source}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default ChatMessage;