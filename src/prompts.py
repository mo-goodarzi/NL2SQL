"""Every prompt string in the project, and nothing else.

Two of the three Step 4 improvements are prompt changes, so isolating them here
means each experiment is a one-file diff you can point at in your write-up.
Nothing in this module calls a model or touches the database -- it only builds
strings, which also makes it trivially testable without spending a token.

``PROMPT_VERSION`` is stamped onto evaluation reports. When you look back at a
trajectory table in three weeks and wonder why v2 beat v1, the version tells
you which prompts produced each row.

A note on the rules below: most of them exist because of a specific, observed
failure mode, and each one is commented with the failure it prevents. Resist
adding rules speculatively. A prompt that lists thirty instructions dilutes the
five that matter, and you will not be able to tell which is doing the work.
"""

from __future__ import annotations

from typing import Any, Sequence

from config import settings

PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Dialect and schema rules
#
# These are SQLite- and Olist-specific on purpose. Generic "you are a SQL
# expert" prompting leaves the model guessing about how dates are stored and
# which column identifies a customer -- and those two guesses account for a
# large share of baseline failures on this dataset.
# ---------------------------------------------------------------------------

DIALECT_RULES = """\
Dialect and storage rules:
- Target SQLite. Use SQLite syntax only.
- Dates are stored as TEXT in 'YYYY-MM-DD HH:MM:SS' form. Extract parts with
  strftime, e.g. strftime('%Y', order_purchase_timestamp) = '2017' for a year,
  or strftime('%Y-%m', ...) for a month. Do not use EXTRACT or DATEPART.
- String concatenation uses ||, not CONCAT.
- Use explicit JOIN ... ON syntax rather than comma joins."""


SCHEMA_RULES = """\
Data model notes:
- order_items is the bridge between orders and both products and sellers. Any
  question relating an order to a product or seller must join through it.
- customers.customer_id is unique per order. A returning shopper has several
  customer_id values but one customer_unique_id, so counting distinct people
  means COUNT(DISTINCT customer_unique_id).
- products.product_category_name is in Portuguese. Join
  product_category_translation to answer questions phrased with English
  category names.
- order_status values are lower-case (e.g. 'delivered', 'canceled').
- order_items.order_item_id is a line-number within the order, not a quantity.
  Count rows to count items."""


OUTPUT_RULES = """\
Output rules:
- Return one SQL statement and nothing else: no explanation, no markdown
  fences, no trailing semicolon commentary.
- Select exactly the columns the question asks for. Avoid SELECT * -- extra
  columns make the answer harder to read and harder to check.
- Qualify every column with a table alias when the query joins more than one
  table, so no column reference is ambiguous.
- Use only tables and columns that appear in the schema below. If the question
  cannot be answered from this schema, return the single word: UNANSWERABLE.
- When the question asks for a "top" or "best" or "worst" N, include both an
  ORDER BY and a LIMIT.
- When aggregating, group by the same expression you select."""


SQL_SYSTEM = f"""\
You are an expert data analyst who writes precise, minimal SQLite queries \
against an e-commerce database. You translate a business question into exactly \
one SQL query.

{DIALECT_RULES}

{SCHEMA_RULES}

{OUTPUT_RULES}"""


EXPLANATION_SYSTEM = """\
You explain query results to a non-technical reader in one short sentence.

Rules:
- Use only the numbers present in the results given to you. Never estimate,
  extrapolate, or add figures that are not shown.
- Lead with the answer itself, not with a description of the query.
- If the result set is empty, say plainly that no rows matched -- do not
  invent a reason or guess at what the user meant.
- One sentence. No preamble, no markdown, no bullet points."""


# ---------------------------------------------------------------------------
# Few-shot examples (Step 4 toggle)
#
# IMPORTANT: none of these may come from eval/testset.yaml. Reusing a test
# question as an example is leakage -- the score stops measuring generalisation
# and starts measuring memorisation, and the number becomes indefensible.
#
# These are chosen to demonstrate the four patterns that fail most often at
# baseline: joining through the bridge table, the Portuguese category join,
# strftime date filtering, and distinct-customer counting. Replace them with
# patterns YOUR eval shows failing -- targeted examples move the number,
# decorative ones just cost tokens.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "How many distinct customers placed an order in 2018?",
        "sql": (
            "SELECT COUNT(DISTINCT c.customer_unique_id) AS customer_count\n"
            "FROM orders o\n"
            "JOIN customers c ON c.customer_id = o.customer_id\n"
            "WHERE strftime('%Y', o.order_purchase_timestamp) = '2018'"
        ),
    },
    {
        "question": "What is the total revenue per English product category?",
        "sql": (
            "SELECT t.product_category_name_english AS category,\n"
            "       SUM(oi.price) AS revenue\n"
            "FROM order_items oi\n"
            "JOIN products p ON p.product_id = oi.product_id\n"
            "JOIN product_category_translation t\n"
            "  ON t.product_category_name = p.product_category_name\n"
            "GROUP BY t.product_category_name_english\n"
            "ORDER BY revenue DESC"
        ),
    },
    {
        "question": "Which five sellers shipped the most items?",
        "sql": (
            "SELECT oi.seller_id, COUNT(*) AS item_count\n"
            "FROM order_items oi\n"
            "GROUP BY oi.seller_id\n"
            "ORDER BY item_count DESC\n"
            "LIMIT 5"
        ),
    },
    {
        "question": "What was the average review score by month in 2017?",
        "sql": (
            "SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,\n"
            "       AVG(r.review_score) AS avg_score\n"
            "FROM order_reviews r\n"
            "JOIN orders o ON o.order_id = r.order_id\n"
            "WHERE strftime('%Y', o.order_purchase_timestamp) = '2017'\n"
            "GROUP BY month\n"
            "ORDER BY month"
        ),
    },
]


def format_examples(examples: Sequence[dict[str, str]] | None = None) -> str:
    """Render few-shot examples as a prompt block, or an empty string."""
    examples = FEW_SHOT_EXAMPLES if examples is None else examples
    if not examples:
        return ""
    blocks = [
        f"Question: {ex['question']}\nSQL:\n{ex['sql']}" for ex in examples
    ]
    return "Examples of correct queries:\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_sql_prompt(
    question: str,
    schema_text: str,
    examples: Sequence[dict[str, str]] | None = None,
) -> str:
    """The first-attempt user message: schema, optional examples, question.

    Args:
        question: The user's natural-language question.
        schema_text: Rendered schema from ``schema.py`` -- either the full
            database or the retrieved subset, depending on
            ``settings.use_schema_retrieval``.
        examples: Overrides the default few-shot block. Pass an empty list to
            force examples off regardless of config; leave as None to follow
            ``settings.use_few_shot``.

    The schema goes before the question deliberately. Providers cache on shared
    prompt prefixes, and the schema is the large, unchanging part -- putting it
    first means every question after the first can hit that cache.
    """
    parts = [f"Database schema:\n\n{schema_text}"]

    if examples is None:
        examples = FEW_SHOT_EXAMPLES if settings.use_few_shot else []
    block = format_examples(examples)
    if block:
        parts.append(block)

    parts.append(f"Question: {question}\n\nSQL:")
    return "\n\n".join(parts)


def build_correction_prompt(
    question: str,
    failed_sql: str,
    error: str,
    schema_text: str,
    history: Sequence[dict[str, Any]] | None = None,
) -> str:
    """The retry message: what was tried, what broke, and try again.

    This is the prompt that makes the self-correction loop work, so it carries
    three things the first-attempt prompt cannot: the exact SQL that failed,
    the database's own error text (enriched with a did-you-mean hint by
    ``db.py``), and -- when there have been several attempts -- the queries
    already ruled out.

    Passing the history matters more than it looks. Without it a model will
    cheerfully produce the same broken query twice, burning the retry budget
    on an attempt it has already made.
    """
    parts = [f"Database schema:\n\n{schema_text}", f"Question: {question}"]

    prior = [h for h in (history or []) if h.get("sql") and h["sql"] != failed_sql]
    if prior:
        lines = "\n\n".join(
            f"Attempt {i + 1}:\n{h['sql']}\nFailed with: {h.get('error', 'unknown error')}"
            for i, h in enumerate(prior)
        )
        parts.append(f"Earlier attempts that also failed -- do not repeat them:\n\n{lines}")

    parts.append(
        f"Your most recent query:\n{failed_sql}\n\n"
        f"It failed with this database error:\n{error}"
    )
    parts.append(
        "Diagnose the cause and write a corrected query. Change what caused "
        "the failure -- do not resubmit the same query. Return only the "
        "corrected SQL."
    )
    return "\n\n".join(parts)


def build_empty_result_prompt(
    question: str,
    executed_sql: str,
    schema_text: str,
) -> str:
    """Retry message for a query that ran fine but returned nothing.

    Zero rows is a different failure from an error, and deserves a different
    prompt. Sometimes zero rows is the correct answer; sometimes the model
    filtered on a value that does not exist in the data (the classic case here
    being an English category name compared against the Portuguese column).
    Telling the model which situation to consider is more useful than feeding
    it a generic 'try again'.
    """
    return (
        f"Database schema:\n\n{schema_text}\n\n"
        f"Question: {question}\n\n"
        f"This query executed successfully but returned no rows:\n{executed_sql}\n\n"
        "Decide which is true:\n"
        "1. No rows genuinely match, and the query is correct.\n"
        "2. A filter is wrong -- a misspelled literal, the wrong case, an "
        "English value compared against a Portuguese column, or a date range "
        "outside the data.\n\n"
        "If (1), return the same query unchanged. If (2), return a corrected "
        "query. Return only SQL."
    )


def render_rows_for_prompt(
    columns: Sequence[str] | None,
    rows: Sequence[Sequence[Any]],
    max_rows: int = 20,
) -> str:
    """Compact result rendering for the explanation prompt.

    Capped because a 100-row result would dominate the prompt and cost more
    than the answer is worth -- the explanation only needs the shape and the
    headline figures.
    """
    if not columns:
        return "(no columns)"
    if not rows:
        return "(no rows)"

    head = " | ".join(str(c) for c in columns)
    body = "\n".join(
        " | ".join("NULL" if v is None else str(v) for v in row)
        for row in rows[:max_rows]
    )
    more = f"\n... and {len(rows) - max_rows} more row(s)" if len(rows) > max_rows else ""
    return f"{head}\n{body}{more}"


def build_explanation_prompt(
    question: str,
    columns: Sequence[str] | None,
    rows: Sequence[Sequence[Any]],
    max_rows: int = 20,
) -> str:
    """Turn a successful result into one plain sentence.

    Note what is deliberately absent: the SQL. The explanation should describe
    what the data says, not what the query did, and including the query tempts
    the model into narrating joins instead of answering the question.
    """
    return (
        f"Question: {question}\n\n"
        f"Query results:\n{render_rows_for_prompt(columns, rows, max_rows)}\n\n"
        "Answer the question in one sentence using only these results."
    )


def prompt_metadata() -> dict[str, Any]:
    """Prompt configuration, for stamping onto evaluation reports and traces."""
    return {
        "prompt_version": PROMPT_VERSION,
        "use_few_shot": settings.use_few_shot,
        "few_shot_count": len(FEW_SHOT_EXAMPLES) if settings.use_few_shot else 0,
    }