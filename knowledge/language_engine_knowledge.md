# gAi Language Engine Knowledge Base

This file records general language-processing capabilities learned from public NLP projects. It is a design/knowledge layer, not a copy of third-party source code or model weights.

## Core capabilities

1. Text normalization
- Normalize Unicode and Bengali text before analysis.
- Preserve meaningful punctuation and sentence boundaries.
- Detect Bengali, English and mixed Banglish input.

2. Tokenization
- Split text into words/subwords and sentences.
- Treat Bengali punctuation and Unicode correctly.
- Keep numbers, names and mixed-script tokens identifiable.

3. Morphology and grammatical structure
- Identify approximate word roles and grammatical features.
- Support POS tagging concepts such as noun, verb, adjective, pronoun, adverb, postposition and conjunction.
- Track tense/person where enough evidence exists.

4. Named entities
- Recognize people, places, organizations, dates, numbers and other named entities.
- Do not treat every capitalized or uncommon token as an entity.

5. Spelling and text correction
- Detect likely Bengali spelling errors.
- Suggest corrections using edit distance, dictionary frequency and context.
- Do not silently change uncertain names or technical terms.

6. Semantic understanding
- Compare meaning rather than exact wording.
- Support intent detection, paraphrase recognition, textual entailment and contradiction detection.
- Use confidence and uncertainty instead of pretending an uncertain match is exact.

7. Question understanding
- Detect question type and extract the main subject/topic.
- Distinguish factual questions, how-to questions, clarification requests, conversational messages and teaching messages.

8. Sentiment and emotion
- Detect approximate positive/negative/neutral sentiment.
- Detect emotions such as frustration, sadness, gratitude and excitement when evidence is sufficient.
- Emotion detection informs conversational behavior; it must not be treated as a diagnosis.

9. Context
- Preserve conversation context when available.
- Resolve references such as "এটা", "ওটা", "সে", "আগেরটা" using recent context.
- Ask for clarification when context is insufficient.

10. Language generation guidance
- Match the user's language unless the user asks otherwise.
- If the user writes Banglish, understand it without forcing unnecessary code-switching in the response.
- Respect formal/informal pronouns and context.
- Prefer clear, concise sentences.

## Project-derived capability map

BanglaBERT demonstrates downstream Bangla NLU tasks including sequence classification, NLI, token classification and question answering. gAi should treat these as capability categories rather than copying the implementation. See: https://github.com/csebuetnlp/banglabert

BNLP provides Bengali tokenization, embeddings, POS tagging, NER, text cleaning, spell checking and language detection. These categories define useful Language Engine modules. See: https://github.com/sagorbrur/bnlp

BanglaNLP resources cover datasets/tasks including POS tagging, lemmatization, NER, punctuation restoration, machine translation, sentiment, emotion, authorship attribution and news categorization. Use these as a map of possible future capabilities, subject to each dataset's license. See: https://github.com/banglanlp/bnlp-resources

Context-sensitive Bangla spell-checking projects demonstrate combining dictionaries, edit distance and contextual language models. gAi may request a dedicated contextual spell-checking capability when its local rules are insufficient. See: https://github.com/MahirMahbub/Contextual-Spell-Checker-For-Bangla

## Growth rule

The Language Engine must not pretend that a capability exists merely because its name appears here. A capability is considered operational only when a human-provided implementation exists and passes its tests.

When a required language capability is missing, gAi should report:
- what it understood,
- what it cannot reliably determine,
- the missing capability,
- the exact file path requested for the human implementation.

## Important licensing rule

Do not copy third-party repository source code, model weights or datasets into gAi merely because they are public. Inspect their licenses and use only material permitted by the applicable license. This knowledge file records high-level capability concepts and project references only.
