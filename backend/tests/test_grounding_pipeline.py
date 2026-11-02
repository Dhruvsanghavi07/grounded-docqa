from docqa.grounding import check, sentences
from docqa.llm import ExtractiveGenerator, build_prompt
from docqa.pipeline import QAPipeline

P = [
    "Errors\nRaise HTTPException with status code 404 when an item is missing.",
    "CORS\nAdd `CORSMiddleware` and allow http://localhost:5173.",
]


def test_supported_answer_passes():
    r = check("Raise an exception with status code 404 [1]. Use `CORSMiddleware` for origins [2].", P)
    assert r.grounded and r.cited == [1, 2]


def test_uncited_sentence_fails():
    assert not check("Raise status code 404.", P).grounded


def test_citation_to_missing_passage_fails():
    r = check("Use status code 404 [7].", P)
    assert not r.grounded and "not retrieved" in r.violations[0]


def test_invented_number_fails():
    r = check("Return status code 418 [1].", P)
    assert not r.grounded and "418" in r.violations[0]


def test_invented_identifier_fails():
    r = check("Use `TrustedHostMiddleware` [2].", P)
    assert not r.grounded and "TrustedHostMiddleware" in r.violations[0]


def test_refusal_needs_no_citation():
    assert check("The documentation provided does not answer this.", P).grounded


def test_sentence_split_keeps_trailing_citation():
    assert sentences("First claim [1]. Second claim [2].") == ["First claim [1].", "Second claim [2]."]


def test_prompt_numbers_passages():
    assert "[1] Errors" in build_prompt("q", P) and "[2] CORS" in build_prompt("q", P)


def test_extractive_quotes_top_passage_and_is_grounded():
    out = ExtractiveGenerator().generate("status code when an item is missing", P)
    assert "[1]" in out and check(out, P).grounded


def test_extractive_refuses_off_topic():
    assert "does not answer" in ExtractiveGenerator().generate("capital of France", P)


class Scripted:
    """Fake LLM that returns canned answers in order, so retry/fallback logic is testable offline."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def generate(self, question, passages):
        self.calls.append(question)
        return self.answers.pop(0)


class Broken:
    name = "broken"

    def generate(self, question, passages):
        raise TimeoutError("model timed out")


def test_good_llm_answer_is_kept(index):
    llm = Scripted("You can define background tasks to be run after returning a response [1].")
    a = QAPipeline(index, llm, k=3).ask("send an email after the response")
    assert a.grounded and not a.fell_back and a.generator == "scripted" and a.attempts == 1


def test_bad_answer_is_retried_with_feedback(index):
    llm = Scripted("It returns status 500 [1].", "Raise `HTTPException` with a status code of 404 [1].")
    a = QAPipeline(index, llm, k=3).ask("How do I return a 404 when an item is not found?")
    assert a.grounded and a.attempts == 2 and not a.fell_back
    assert "broke the citation rules" in llm.calls[1]


def test_falls_back_when_retries_fail(index):
    llm = Scripted("Use status 999 [1].", "Still 999 [1].")
    a = QAPipeline(index, llm, k=3).ask("How do I return a 404 when an item is not found?")
    assert a.fell_back and a.generator == "extractive" and a.grounded and a.violations


def test_falls_back_on_generator_error(index):
    a = QAPipeline(index, Broken(), k=3).ask("CORS for my frontend origin")
    assert a.fell_back and "TimeoutError" in a.violations[0]


def test_answer_reports_sources_and_timings(index):
    a = QAPipeline(index, k=2).ask("CORS origins")
    assert len(a.sources) == 2 and a.sources[0].n == 1 and a.retrieval_ms >= 0
