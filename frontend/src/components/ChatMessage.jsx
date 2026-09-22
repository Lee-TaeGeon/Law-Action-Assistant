import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import SourceCard from "./SourceCard";

function normalizeMarkdown(text = "") {
  return text
    .replace(
      /\\([#*_`~])/g,
      "$1"
    )
    .replace(
      /<br\s*\/?>/gi,
      "\n"
    );
}

function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="message user-message">
        <div className="message-content">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="message assistant-message">
      <div className="message-avatar">
        AI
      </div>

      <div className="message-content">
        {message.category && (
          <span className="category-badge">
            {message.category}
          </span>
        )}

        <div className="markdown-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
          >
            {normalizeMarkdown(
              message.content
            )}
          </ReactMarkdown>
        </div>

        {message.sources?.length > 0 && (
          <div className="sources">
            <p className="sources-title">
              관련 법령
            </p>

            {message.sources.map(
              (source, index) => (
                <SourceCard
                  key={`${source.law_name}-${source.article || index}`}
                  source={source}
                />
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatMessage;