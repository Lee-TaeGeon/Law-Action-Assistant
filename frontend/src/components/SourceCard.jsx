function SourceCard({ source }) {
  return (
    <details className="source-card">
      <summary>
        <span>⚖</span>

        <strong>
          {source.law_name}
          {source.article && ` ${source.article}`}
        </strong>

        <span className="source-toggle">
          내용 보기
        </span>
      </summary>

      <div className="source-content">
        {source.content}
      </div>
    </details>
  );
}

export default SourceCard;