import streamlit as st
import pandas as pd
import itertools
import string
from io import BytesIO
from wordfreq import zipf_frequency

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Verbal Reasoning Bridge Checker", layout="wide")

ALPHABET = string.ascii_lowercase


# =========================================================
# WORD CHECKING
# =========================================================
def is_common_word(word: str, min_zipf: float = 3.5) -> bool:
    """
    Returns True if the word is alphabetic and common enough in English.
    """
    word = str(word).strip().lower()

    if not word.isalpha():
        return False

    return zipf_frequency(word, "en") >= min_zipf


def generate_letter_combos(n: int):
    """
    Generates all possible lowercase letter combinations of length n.
    Example for n=2: aa, ab, ac, ... zz
    """
    for combo in itertools.product(ALPHABET, repeat=n):
        yield "".join(combo)


# =========================================================
# PATTERN HANDLING
# =========================================================
def infer_missing_letters(left_fragment: str, right_fragment: str):
    """
    Infers the bridge length from:
    - trailing underscores on the left fragment
    - leading underscores on the right fragment

    Example:
    left  = 'phot_'
    right = '_pen'
    => 1

    Example:
    left  = 'bea__'
    right = '__time'
    => 2

    Returns an integer if valid, otherwise None.
    """
    left_fragment = str(left_fragment).strip()
    right_fragment = str(right_fragment).strip()

    left_count = 0
    for ch in reversed(left_fragment):
        if ch == "_":
            left_count += 1
        else:
            break

    right_count = 0
    for ch in right_fragment:
        if ch == "_":
            right_count += 1
        else:
            break

    if left_count == 0 or right_count == 0:
        return None

    if left_count != right_count:
        return None

    return left_count


def clean_fragment(fragment: str) -> str:
    """
    Lowercases and trims spaces.
    """
    return str(fragment).strip().lower()


def build_words(left_fragment: str, right_fragment: str, bridge: str):
    """
    Replaces the trailing underscores in the left fragment
    and leading underscores in the right fragment with bridge.
    """
    n = len(bridge)
    left_word = left_fragment[:-n] + bridge
    right_word = bridge + right_fragment[n:]
    return left_word, right_word


def validate_fragments(left_fragment: str, right_fragment: str, missing_letters: int):
    """
    Validates the fragment format.
    Left must end with exactly missing_letters underscores.
    Right must start with exactly missing_letters underscores.
    """
    left_fragment = clean_fragment(left_fragment)
    right_fragment = clean_fragment(right_fragment)

    if missing_letters < 1:
        return False, "Missing letters must be at least 1."

    if not left_fragment.endswith("_" * missing_letters):
        return False, f"Left fragment must end with exactly {missing_letters} underscore(s)."

    if not right_fragment.startswith("_" * missing_letters):
        return False, f"Right fragment must start with exactly {missing_letters} underscore(s)."

    left_base = left_fragment[:-missing_letters]
    right_base = right_fragment[missing_letters:]

    if not left_base.isalpha():
        return False, "Left fragment contains invalid characters before the gap."

    if not right_base.isalpha():
        return False, "Right fragment contains invalid characters after the gap."

    return True, ""


# =========================================================
# SOLVER
# =========================================================
def find_valid_bridges(
    left_fragment: str,
    right_fragment: str,
    missing_letters: int,
    min_zipf: float = 3.5
):
    """
    Returns all valid bridges where both completed words are common English words.
    """
    left_fragment = clean_fragment(left_fragment)
    right_fragment = clean_fragment(right_fragment)

    valid, message = validate_fragments(left_fragment, right_fragment, missing_letters)
    if not valid:
        return {
            "status": "Format error",
            "error": message,
            "matches": []
        }

    matches = []

    for bridge in generate_letter_combos(missing_letters):
        left_word, right_word = build_words(left_fragment, right_fragment, bridge)

        if is_common_word(left_word, min_zipf) and is_common_word(right_word, min_zipf):
            matches.append({
                "bridge": bridge,
                "left_word": left_word,
                "right_word": right_word,
                "left_zipf": round(zipf_frequency(left_word, "en"), 2),
                "right_zipf": round(zipf_frequency(right_word, "en"), 2),
            })

    if len(matches) == 0:
        status = "No valid answer"
    elif len(matches) == 1:
        status = "Safe"
    else:
        status = "Unsafe: multiple answers"

    return {
        "status": status,
        "error": "",
        "matches": matches
    }


# =========================================================
# CSV PROCESSING
# =========================================================
def process_dataframe(df: pd.DataFrame, min_zipf: float):
    """
    Processes a CSV with columns:
    - left_fragment
    - right_fragment
    Optional:
    - missing_letters

    If missing_letters is absent, the app will try to infer it from underscores.
    """
    required_cols = {"left_fragment", "right_fragment"}
    missing_required = required_cols - set(df.columns)

    if missing_required:
        raise ValueError(
            f"CSV must contain these columns: left_fragment, right_fragment. "
            f"Missing: {', '.join(sorted(missing_required))}"
        )

    results = []

    for _, row in df.iterrows():
        left_fragment = str(row["left_fragment"]).strip()
        right_fragment = str(row["right_fragment"]).strip()

        if "missing_letters" in df.columns and pd.notna(row["missing_letters"]):
            try:
                missing_letters = int(row["missing_letters"])
            except Exception:
                missing_letters = None
        else:
            missing_letters = infer_missing_letters(left_fragment, right_fragment)

        if missing_letters is None:
            results.append({
                "left_fragment": left_fragment,
                "right_fragment": right_fragment,
                "missing_letters": "",
                "status": "Format error",
                "number_of_answers": 0,
                "answers": "",
                "completed_left_words": "",
                "completed_right_words": "",
                "error": "Could not determine missing letters. Add a missing_letters column or use matching underscores."
            })
            continue

        result = find_valid_bridges(
            left_fragment=left_fragment,
            right_fragment=right_fragment,
            missing_letters=missing_letters,
            min_zipf=min_zipf
        )

        matches = result["matches"]

        results.append({
            "left_fragment": left_fragment,
            "right_fragment": right_fragment,
            "missing_letters": missing_letters,
            "status": result["status"],
            "number_of_answers": len(matches),
            "answers": ", ".join(m["bridge"] for m in matches),
            "completed_left_words": ", ".join(m["left_word"] for m in matches),
            "completed_right_words": ", ".join(m["right_word"] for m in matches),
            "error": result["error"]
        })

    return pd.DataFrame(results)


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# UI
# =========================================================
st.title("Verbal Reasoning Bridge Checker")
st.write(
    "Check whether a letter or group of letters can end one word and begin the next, "
    "and flag questions with multiple possible answers."
)

with st.expander("How to format questions", expanded=False):
    st.markdown(
        """
**Single question examples**
- `phot_` and `_pen`
- `bea__` and `__time`

**CSV columns**
- `left_fragment`
- `right_fragment`
- optional: `missing_letters`

If `missing_letters` is not included, the app will try to infer it from the underscores.
        """
    )

col1, col2 = st.columns([1, 1])

with col1:
    mode = st.radio(
        "Choose mode",
        ["Single question", "Batch CSV"],
        horizontal=True
    )

with col2:
    min_zipf = st.slider(
        "Minimum word frequency threshold",
        min_value=1.0,
        max_value=6.0,
        value=3.5,
        step=0.1,
        help="Higher = stricter, fewer obscure words allowed."
    )

st.divider()

# =========================================================
# SINGLE QUESTION MODE
# =========================================================
if mode == "Single question":
    st.subheader("Single question checker")

    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        left_fragment = st.text_input("Left fragment", value="phot_")

    with c2:
        right_fragment = st.text_input("Right fragment", value="_pen")

    with c3:
        missing_letters_input = st.number_input(
            "Missing letters",
            min_value=1,
            max_value=3,
            value=1,
            step=1
        )

    auto_infer = st.checkbox("Auto-infer missing letters from underscores", value=True)

    if st.button("Check question"):
        if auto_infer:
            inferred = infer_missing_letters(left_fragment, right_fragment)
            missing_letters = inferred if inferred is not None else int(missing_letters_input)
        else:
            missing_letters = int(missing_letters_input)

        result = find_valid_bridges(
            left_fragment=left_fragment,
            right_fragment=right_fragment,
            missing_letters=missing_letters,
            min_zipf=min_zipf
        )

        st.markdown(f"**Status:** {result['status']}")

        if result["error"]:
            st.error(result["error"])
        else:
            matches = result["matches"]

            st.write(f"**Number of valid answers:** {len(matches)}")

            if matches:
                display_df = pd.DataFrame(matches)
                display_df = display_df.rename(columns={
                    "bridge": "answer",
                    "left_word": "completed left word",
                    "right_word": "completed right word",
                    "left_zipf": "left zipf",
                    "right_zipf": "right zipf"
                })
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No valid answers found at this threshold.")

# =========================================================
# BATCH CSV MODE
# =========================================================
else:
    st.subheader("Batch CSV checker")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown(
        """
**Expected columns**
- `left_fragment`
- `right_fragment`
- optional: `missing_letters`
        """
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("Preview of uploaded CSV:")
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("Check CSV"):
                results_df = process_dataframe(df, min_zipf=min_zipf)

                st.success("Finished checking CSV.")
                st.dataframe(results_df, use_container_width=True)

                safe_count = (results_df["status"] == "Safe").sum()
                unsafe_count = (results_df["status"] == "Unsafe: multiple answers").sum()
                no_answer_count = (results_df["status"] == "No valid answer").sum()
                format_error_count = (results_df["status"] == "Format error").sum()

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Safe", int(safe_count))
                s2.metric("Unsafe", int(unsafe_count))
                s3.metric("No valid answer", int(no_answer_count))
                s4.metric("Format errors", int(format_error_count))

                csv_bytes = dataframe_to_csv_download(results_df)
                st.download_button(
                    label="Download checked CSV",
                    data=csv_bytes,
                    file_name="checked_bridge_questions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not read/process CSV: {e}")
