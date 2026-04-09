import streamlit as st
import pandas as pd
import itertools
import string
from wordfreq import zipf_frequency

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Verbal Reasoning Checker", layout="wide")

ALPHABET = string.ascii_lowercase


# =========================================================
# WORD CHECKING
# =========================================================
def is_common_word(word: str, min_zipf: float = 3.5) -> bool:
    word = str(word).strip().lower()
    if not word.isalpha():
        return False
    return zipf_frequency(word, "en") >= min_zipf


def generate_letter_combos(n: int):
    for combo in itertools.product(ALPHABET, repeat=n):
        yield "".join(combo)


# =========================================================
# GENERAL HELPERS
# =========================================================
def clean_fragment(fragment: str) -> str:
    return str(fragment).strip().lower()


# =========================================================
# BRIDGE CHECKER HELPERS
# =========================================================
def infer_missing_letters(left_fragment: str, right_fragment: str):
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


def build_words(left_fragment: str, right_fragment: str, bridge: str):
    n = len(bridge)
    left_word = left_fragment[:-n] + bridge
    right_word = bridge + right_fragment[n:]
    return left_word, right_word


def validate_fragments(left_fragment: str, right_fragment: str, missing_letters: int):
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


def find_valid_bridges(left_fragment: str, right_fragment: str, missing_letters: int, min_zipf: float = 3.5):
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
# SINGLE WORD CHECKER HELPERS
# =========================================================
def infer_single_word_missing_letters(fragment: str):
    fragment = clean_fragment(fragment)
    count = 0

    for ch in reversed(fragment):
        if ch == "_":
            count += 1
        else:
            break

    return count if count > 0 else None


def validate_single_word_fragment(fragment: str, missing_letters: int):
    fragment = clean_fragment(fragment)

    if missing_letters < 1:
        return False, "Missing letters must be at least 1."

    if not fragment.endswith("_" * missing_letters):
        return False, f"Fragment must end with exactly {missing_letters} underscore(s)."

    base = fragment[:-missing_letters]

    if not base.isalpha():
        return False, "Fragment contains invalid characters before the gap."

    return True, ""


def find_single_word_completions(fragment: str, missing_letters: int, min_zipf: float = 3.5):
    fragment = clean_fragment(fragment)

    valid, message = validate_single_word_fragment(fragment, missing_letters)
    if not valid:
        return {
            "status": "Format error",
            "error": message,
            "matches": []
        }

    matches = []

    for ending in generate_letter_combos(missing_letters):
        word = fragment[:-missing_letters] + ending

        if is_common_word(word, min_zipf):
            matches.append({
                "ending": ending,
                "word": word,
                "zipf": round(zipf_frequency(word, "en"), 2)
            })

    if len(matches) == 0:
        status = "No valid completions"
    elif len(matches) == 1:
        status = "1 valid completion"
    else:
        status = f"{len(matches)} valid completions"

    return {
        "status": status,
        "error": "",
        "matches": matches
    }


# =========================================================
# CSV PROCESSING - BRIDGE
# =========================================================
def process_bridge_dataframe(df: pd.DataFrame, min_zipf: float):
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


# =========================================================
# CSV PROCESSING - SINGLE WORD
# =========================================================
def process_single_word_dataframe(df: pd.DataFrame, min_zipf: float):
    required_cols = {"fragment"}
    missing_required = required_cols - set(df.columns)

    if missing_required:
        raise ValueError(
            f"CSV must contain this column: fragment. "
            f"Missing: {', '.join(sorted(missing_required))}"
        )

    results = []

    for _, row in df.iterrows():
        fragment = str(row["fragment"]).strip()

        if "missing_letters" in df.columns and pd.notna(row["missing_letters"]):
            try:
                missing_letters = int(row["missing_letters"])
            except Exception:
                missing_letters = None
        else:
            missing_letters = infer_single_word_missing_letters(fragment)

        if missing_letters is None:
            results.append({
                "fragment": fragment,
                "missing_letters": "",
                "status": "Format error",
                "number_of_completions": 0,
                "endings": "",
                "completed_words": "",
                "zipf_scores": "",
                "error": "Could not determine missing letters. Add a missing_letters column or use trailing underscores."
            })
            continue

        result = find_single_word_completions(
            fragment=fragment,
            missing_letters=missing_letters,
            min_zipf=min_zipf
        )

        matches = result["matches"]

        results.append({
            "fragment": fragment,
            "missing_letters": missing_letters,
            "status": result["status"],
            "number_of_completions": len(matches),
            "endings": ", ".join(m["ending"] for m in matches),
            "completed_words": ", ".join(m["word"] for m in matches),
            "zipf_scores": ", ".join(str(m["zipf"]) for m in matches),
            "error": result["error"]
        })

    return pd.DataFrame(results)


# =========================================================
# UI
# =========================================================
st.title("Verbal Reasoning Word Checker")

st.write(
    "Check bridge questions, batch-check bridge CSVs, check single word endings, or batch-check single word fragments."
)

col1, col2 = st.columns([2, 1])

with col1:
    mode = st.radio(
        "Choose mode",
        [
            "Bridge checker: single question",
            "Bridge checker: batch CSV",
            "Single word ending checker",
            "Single word ending checker: batch CSV"
        ],
        horizontal=True
    )

with col2:
    min_zipf = st.slider(
        "Minimum word frequency threshold",
        min_value=1.0,
        max_value=6.0,
        value=3.5,
        step=0.1,
        help="Higher = stricter. Words must meet this threshold."
    )

st.divider()

# =========================================================
# MODE 1: BRIDGE CHECKER SINGLE
# =========================================================
if mode == "Bridge checker: single question":
    st.subheader("Bridge checker: single question")

    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        left_fragment = st.text_input("Left fragment", value="phot_")

    with c2:
        right_fragment = st.text_input("Right fragment", value="_pen")

    with c3:
        missing_letters_input = st.number_input(
            "Missing letters",
            min_value=1,
            max_value=4,
            value=1,
            step=1
        )

    auto_infer = st.checkbox("Auto-infer missing letters from underscores", value=True)

    if st.button("Check bridge question"):
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
                display_df = pd.DataFrame(matches).rename(columns={
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
# MODE 2: BRIDGE CHECKER BATCH CSV
# =========================================================
elif mode == "Bridge checker: batch CSV":
    st.subheader("Bridge checker: batch CSV")

    uploaded_file = st.file_uploader("Upload bridge CSV", type=["csv"], key="bridge_csv")

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

            if st.button("Check bridge CSV"):
                results_df = process_bridge_dataframe(df, min_zipf=min_zipf)

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

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download checked bridge CSV",
                    data=csv_bytes,
                    file_name="checked_bridge_questions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not read/process CSV: {e}")

# =========================================================
# MODE 3: SINGLE WORD ENDING CHECKER
# =========================================================
elif mode == "Single word ending checker":
    st.subheader("Single word ending checker")

    c1, c2 = st.columns([3, 1])

    with c1:
        fragment = st.text_input("Word fragment", value="phot_")

    with c2:
        missing_letters_input = st.number_input(
            "Missing letters",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            key="single_word_missing_letters"
        )

    auto_infer_single = st.checkbox(
        "Auto-infer missing letters from underscores",
        value=True,
        key="auto_single"
    )

    if st.button("Check word endings"):
        if auto_infer_single:
            inferred = infer_single_word_missing_letters(fragment)
            missing_letters = inferred if inferred is not None else int(missing_letters_input)
        else:
            missing_letters = int(missing_letters_input)

        result = find_single_word_completions(
            fragment=fragment,
            missing_letters=missing_letters,
            min_zipf=min_zipf
        )

        st.markdown(f"**Status:** {result['status']}")

        if result["error"]:
            st.error(result["error"])
        else:
            matches = result["matches"]
            st.write(f"**Number of valid completions:** {len(matches)}")

            if matches:
                display_df = pd.DataFrame(matches).rename(columns={
                    "ending": "ending",
                    "word": "completed word",
                    "zipf": "zipf"
                })
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No valid completions found at this threshold.")

# =========================================================
# MODE 4: SINGLE WORD ENDING CHECKER BATCH CSV
# =========================================================
else:
    st.subheader("Single word ending checker: batch CSV")

    uploaded_file = st.file_uploader("Upload single-word CSV", type=["csv"], key="single_word_csv")

    st.markdown(
        """
**Expected columns**
- `fragment`
- optional: `missing_letters`

**Example**
- `phot_`
- `bea__`
- `tabl_`
        """
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded CSV:")
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("Check single-word CSV"):
                results_df = process_single_word_dataframe(df, min_zipf=min_zipf)

                st.success("Finished checking CSV.")
                st.dataframe(results_df, use_container_width=True)

                format_error_count = (results_df["status"] == "Format error").sum()
                no_completion_count = (results_df["number_of_completions"] == 0).sum()
                one_completion_count = (results_df["number_of_completions"] == 1).sum()
                multiple_completion_count = (results_df["number_of_completions"] > 1).sum()

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("1 completion", int(one_completion_count))
                s2.metric("Multiple completions", int(multiple_completion_count))
                s3.metric("No completions", int(no_completion_count))
                s4.metric("Format errors", int(format_error_count))

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download checked single-word CSV",
                    data=csv_bytes,
                    file_name="checked_single_word_fragments.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not read/process CSV: {e}")
