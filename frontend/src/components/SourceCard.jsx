function SourceCard({ source }) {
  return (
    <details className="source-card">
      <summary>
        <span>⚖️</span>

        <strong>
          {source.law_name}

          {source.article &&
            ` ${source.article}`}
        </strong>

        <span className="source-toggle">
          내용 보기
        </span>
      </summary>

      <div className="source-card-content">
        {source.article_title && (
          <strong>
            {source.article_title}
          </strong>
        )}

        {source.article_title && (
          <br />
        )}

        {source.content ||
          "법령 내용이 없습니다."}
      </div>
    </details>
  );
}

export default SourceCard;