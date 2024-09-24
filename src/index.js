import {getEncoding} from "js-tiktoken";
import {TokenTextSplitter} from "@langchain/textsplitters";
import {ChatOpenAI} from "@langchain/openai";
import {StringOutputParser} from "@langchain/core/output_parsers";
import {BaseChatModel} from "@langchain/core/language_models/chat_models";

const MAX_TOKEN_PER_CHUNK = 1000
const DEFAULT_ENCODING_NAME = 'cl100k_base'

/**
 * Translate the source_text from source_lang to target_lang.
 * @param {string} source_lang
 * @param {string} target_lang
 * @param {string} source_text
 * @param {string} country
 * @param {number} max_tokens
 * @return {Promise<string>} translation
 */
export default async function translate(source_lang, target_lang, source_text, country, model = null, max_tokens = MAX_TOKEN_PER_CHUNK) {
    if (model === null) {
        model = getDefaultModel()
    }
    const num_tokens_in_text = num_tokens_in_string(source_text)
    console.log(num_tokens_in_text)
    if (num_tokens_in_text < max_tokens) {
        return one_chunk_translate_text(model, source_lang, target_lang, source_text, country);
    }
    const token_size = calculate_chunk_size(num_tokens_in_text, max_tokens)
    const text_splitter = new TokenTextSplitter({
        encodingName: DEFAULT_ENCODING_NAME,
        chunkSize: token_size,
        chunkOverlap: 0
    })
    const source_text_chunks = await text_splitter.splitText(source_text)
    const translation_2_chunks = await multi_chunk_translation(model, source_lang, target_lang, source_text_chunks, country)
    return translation_2_chunks.join("")
}

function getDefaultModel() {
    return new ChatOpenAI({
        model: 'gpt-4o',
        apiKey: process.env.OPENAI_API_KEY
    })
}

function num_tokens_in_string(input_str, encoding_name = DEFAULT_ENCODING_NAME) {
    const enc = getEncoding(encoding_name);
    return enc.encode(input_str).length;
}

function calculate_chunk_size(token_count, token_limit) {
    //todo: 类型对齐
    if (token_count <= token_limit) {
        return token_count
    }
    const num_chunks = (token_count + token_limit - 1) / token_limit
    let chunk_size = token_count / num_chunks
    const remaining_tokens = token_count % token_limit
    if (remaining_tokens > 0) {
        chunk_size += remaining_tokens / num_chunks
    }
    return chunk_size
}

/**
 * Translate a single chunk of text from the source language to the target language.
 *
 * This function performs a two-step translation process:
 *     1. Get an initial translation of the source text.
 *     2. Reflect on the initial translation and generate an improved translation.
 * @param {BaseChatModel} model LLM model.
 * @param source_lang The source language of the text.
 * @param target_lang The target language for the translation.
 * @param source_text The text to be translated.
 * @param country Country specified for the target language.
 * @return {string} The improved translation of the source text.
 */
async function one_chunk_translate_text(model, source_lang, target_lang, source_text, country) {
    const translation1 = await one_chunk_initial_translation(model, source_lang, target_lang, source_text)
    const reflection = await one_chunk_reflect_on_translation(model, source_lang, target_lang, source_text, translation1, country)
    return await one_chunk_improve_translation(model, source_lang, target_lang, source_text, translation1, reflection)
}

/**
 * Improves the translation of multiple text chunks based on the initial translation and reflection.
 * @param {BaseChatModel} model LLM model.
 * @param {string} source_lang
 * @param {string} target_lang
 * @param {string[]} source_text_chunks
 * @param {string} country
 * @return {Promise<string[]>}
 */
async function multi_chunk_translation(model, source_lang, target_lang, source_text_chunks, country = "") {
    const translation_1_chunks = await multi_chunk_initial_translation(model, source_lang, target_lang, source_text_chunks)
    return translation_1_chunks
    const reflection_chunks = await multi_chunk_reflect_on_translation(model, source_lang, target_lang, source_text_chunks, translation_1_chunks, country)
    return await multi_chunk_improve_translation(model, source_lang, target_lang, source_text_chunks, translation_1_chunks, reflection_chunks)
}

/**
 * Translate the entire text as one chunk using an LLM.
 * @param {BaseChatModel} model LLM model.
 * @param {string} source_lang The source language of the text.
 * @param {string} target_lang The target language for translation.
 * @param {string} source_text The text to be translated.
 * @return {Promise<string>} The translated text.
 */
async function one_chunk_initial_translation(model, source_lang, target_lang, source_text) {
    const system_message = `You are an expert linguist, specializing in translation from ${source_lang} to ${target_lang}.`
    const translation_prompt = `This is an ${source_lang} to ${target_lang} translation, please provide the ${target_lang} translation for this text.
Do not provide any explanations or text apart from the translation.
${source_lang}: ${source_text}

${target_lang}:`
    const result = await model.invoke([
        ["system", system_message],
        ["user", translation_prompt]
    ], {})
    const parser = new StringOutputParser()
    return parser.invoke(result)
}

/**
 * Use an LLM to reflect on the translation, treating the entire text as one chunk.
 * @param {BaseChatModel} model LLM model.
 * @param {string} source_lang The source language of the text.
 * @param {string} target_lang The target language of the translation.
 * @param {string} source_text The original text in the source language.
 * @param {string} translation_1 The initial translation of the source text.
 * @param {string} country Country specified for the target language.
 * @return {Promise<string>} The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
 */
async function one_chunk_reflect_on_translation(model, source_lang, target_lang, source_text, translation_1, country = "") {
    const system_message = `You are an expert linguist specializing in translation from ${source_lang} to ${target_lang}.
You will be provided with a source text and its translation and your goal is to improve the translation.`;
    let reflection_prompt = `Your task is to carefully read a source text and a translation from ${source_lang} to ${target_lang}, and then give constructive criticisms and helpful suggestions to improve the translation.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
${source_text}
</SOURCE_TEXT>

<TRANSLATION>
${translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying ${target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms ${target_lang}).

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.`
    if (country !== "") {
        reflection_prompt = `Your task is to carefully read a source text and a translation from ${source_lang} to ${target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \\
The final style and tone of the translation should match the style of ${target_lang} colloquially spoken in ${country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
${source_text}
</SOURCE_TEXT>

<TRANSLATION>
${translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's 
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying ${target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms ${target_lang}).

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.`
    }
    const result = await model.invoke([
        ["system", system_message],
        ["user", reflection_prompt]
    ])
    const parser = new StringOutputParser()
    return parser.invoke(result)
}

/**
 * Use the reflection to improve the translation, treating the entire text as one chunk.
 * @param {BaseChatModel} model LLM model.
 * @param {string} source_lang The source language of the text.
 * @param {string} target_lang The target language for the translation.
 * @param {string} source_text The original text in the source language.
 * @param {string} translation_1 The initial translation of the source text.
 * @param {string} reflection Expert suggestions and constructive criticism for improving the translation.
 * @return {Promise<string>} The improved translation based on the expert suggestions.
 */
async function one_chunk_improve_translation(model, source_lang, target_lang, source_text, translation_1, reflection) {
    const system_message = `You are an expert linguist, specializing in translation editing from ${source_lang} to ${target_lang}.`
    const prompt = `Your task is to carefully read, then edit, a translation from ${source_lang} to ${target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS>
as follows:

<SOURCE_TEXT>
${source_text}
</SOURCE_TEXT>

<TRANSLATION>
${translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
${reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying ${target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else.`
    const result = await model.invoke([
        ["system", system_message],
        ["user", prompt]
    ])
    const parser = new StringOutputParser()
    return parser.invoke(result)
}

async function multi_chunk_initial_translation(model, source_lang, target_lang, source_text_chunks) {
    const parser = new StringOutputParser()
    const system_message = `You are an expert linguist, specializing in translation from ${source_lang} to ${target_lang}.`
    let translation_chunks = []
    for (let i = 0; i < source_text_chunks.length; i++) {
        const tagged_text = source_text_chunks.slice(0, i).join("")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + source_text_chunks.slice(i + 1).join("");
        const chunk_to_translate = source_text_chunks[i]
        const prompt = `Your task is to provide a professional translation from ${source_lang} to ${target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
${tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
${chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.`
        const result = await model.invoke([
            ["system", system_message],
            ["user", prompt]
        ])
        const translation = await parser.invoke(result)
        console.log(`translation_1.[${i}] = ${translation}`)
        translation_chunks.push(translation)
    }
    return translation_chunks
}

async function multi_chunk_reflect_on_translation(model, source_lang, target_lang, source_text_chunks, translation_1_trunks, country) {
    const system_message = `You are an expert linguist specializing in translation from ${source_lang} to ${target_lang}. 
You will be provided with a source text and its translation and your goal is to improve the translation.`
    const reflection_chunks = []
    const parser = new StringOutputParser()
    for (let i = 0; i < source_text_chunks.length; i++) {
        const tagged_text = source_text_chunks.slice(0, i).join("")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + source_text_chunks.slice(i + 1).join("");
        const chunk_to_translate = source_text_chunks[i]
        const translation_1_chunk = translation_1_trunks[i]
        let prompt = `Your task is to carefully read a source text and part of a translation of that text from ${source_lang} to ${target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
${tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
${chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
${translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms ${target_lang}).

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.`
        if (country != "") {
            prompt = `Your task is to carefully read a source text and part of a translation of that text from ${source_lang} to ${target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of ${target_lang} colloquially spoken in ${country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
${tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
${chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
${translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms ${target_lang}).

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.`
        }
        const result = await model.invoke([
            ['system', system_message],
            ['user', prompt]
        ])
        const reflection = await parser.invoke(result)
        reflection_chunks.push(reflection)
    }
    return reflection_chunks
}

async function multi_chunk_improve_translation(model, source_lang, target_lang, source_text_chunks, translation_1_trunks, reflection_chunks) {
    const system_message = `You are an expert linguist, specializing in translation editing from ${source_lang} to ${target_lang}.`
    const translation_2_trunks = []
    const parser = new StringOutputParser()
    for (let i = 0; i < source_text_chunks.length; i++) {
        const tagged_text = source_text_chunks.slice(0, i).join("")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + source_text_chunks.slice(i + 1).join("");
        const chunk_to_translate = source_text_chunks[i]
        const translation_1_chunk = translation_1_trunks[i]
        const reflection_chunk = reflection_chunks[i]
        const prompt = `Your task is to carefully read, then improve, a translation from ${source_lang} to ${target_lang}, taking into
account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
${tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
${chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
${translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, are as follows:
<EXPERT_SUGGESTIONS>
${reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), 
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else.`
        const result = await model.invoke([
            ['system', system_message],
            ['user', prompt]
        ])
        const translation_2 = await parser.invoke(result)
        translation_2_trunks.push(translation_2)
    }
    return translation_2_trunks
}
