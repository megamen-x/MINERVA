# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

CLAIM_EXTRACTION_PROMPT = """
-Целевая деятельность-
Вы - интеллектуальный помощник, который помогает человеку-аналитику анализировать утверждения к определенным сущностям, представленным в текстовом документе.

-Цель-
При наличии текстового документа, потенциально относящегося к данному виду деятельности, и списка типов сущностей, определите все сущности этих типов из текста и все взаимосвязи между идентифицированными сущностями.

-Шаги-
1. Извлеките все именованные сущности, соответствующие заданной спецификации сущности. Спецификация сущности может быть либо списком имен сущностей, либо списком типов сущностей.
2. Для каждой сущности, определенной на шаге 1, извлеките все утверждения, связанные с этой сущностью. утверждения должны соответствовать заданному описанию утверждения, а сущность должна быть предметом утверждения.
Для каждой утверждения извлеките следующую информацию:
- Subject: имя субъекта, являющегося предметом утверждения, с заглавной буквы. Субъект - это тот, кто совершил действие, описанное в утверждения. Субъект должен быть одним из именованных субъектов, определенных на шаге 1.
- Object: название сущности, являющейся объектом утверждения, с заглавной буквы. Объект сущности это объект, который либо сообщает/обрабатывает, либо на который влияют действия, описанные в утверждении. Если объектный объект неизвестен, используйте **НЕИЗВЕСТЕН**.
- Claim Type: укажите общую категорию утверждения с заглавной буквы. Назовите его таким образом, чтобы его можно было повторить в нескольких текстовых сообщениях, чтобы похожие утверждения имели один и тот же тип
- Claim Status: **ВЕРНО**, **ЛОЖНО** или **ПРЕДПОЛОЖИТЕЛЬНО**. "ВЕРНО" означает, что утверждение подтверждено, "ЛОЖНО" означает, что утверждение признано ложным, "ПРЕДПОЛОЖИТЕЛЬНО" означает, что утверждение не подтверждено.
- Claim Description: Подробное описание, объясняющее обоснование утверждения, вместе со всеми соответствующими доказательствами и ссылками.
- Claim Date: Период (start_date, end_date), когда было подано заявление. Как start_date, так и end_date должны быть указаны в формате ISO-8601. Если запрос был сделан на одну дату, а не на диапазон дат, задайте одинаковую дату как для start_date, так и для end_date. Если дата неизвестна, верните **НЕИЗВЕСТНО**.
- Claim Source Text: Список ** всех** цитат из оригинального текста, которые имеют отношение к утверждению.

Отформатируйте каждое утверждение, как (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Верните выходные данные на русском языке в виде единого списка всех утверждений, указанных в шагах 1 и 2. В качестве разделителя списка используйте **{record_delimiter}**.

4. После завершения выведите {completion_delimiter}

-Примеры-
Пример 1:
Entity specification: организация
Claim description: красные флажки, связанные с сущностью
Text: Согласно статье от 2022/01/10, компания А была оштрафована за фальсификацию результатов торгов при участии в нескольких публичных тендерах, опубликованных правительственным агентством В. Компания принадлежит человеку С, которое подозревалось в причастности к коррупционной деятельности в 2015 году.
Output:

(КОМПАНИЯ A{tuple_delimiter}ПРАВИТЕЛЬСТВЕННОЕ АГЕНСТВО B{tuple_delimiter}АНТИКОНКУРЕНТНАЯ ПРАКТИКА{tuple_delimiter}ВЕРНО{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Компания A была уличена в антиконкурентной практике, поскольку была оштрафована за сговор на торгах при участии в нескольких публичных тендерах, опубликованных правительственным агентством B, согласно статье, опубликованной 2022/01/10{tuple_delimiter}Согласно статье, опубликованной 2022/01/10, компания A была оштрафована за сговор на торгах при участии в нескольких публичных тендерах, опубликованных правительственным агентством B.)
{completion_delimiter}

Пример 2:
Entity specification: Компания A, Человек C
Claim description: красные флажки, связанные с сущностью
Text: Согласно статье от 2022/01/10, компания А была оштрафована за фальсификацию результатов торгов при участии в нескольких публичных тендерах, опубликованных правительственным агентством В. Компания принадлежит человеку С, которое подозревалось в причастности к коррупционной деятельности в 2015 году.
Output:

(КОМПАНИЯ A{tuple_delimiter}ПРАВИТЕЛЬСТВЕННОЕ АГЕНСТВО B{tuple_delimiter}АНТИКОНКУРЕНТНАЯ ПРАКТИКА{tuple_delimiter}ВЕРНО{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Компания A была уличена в антиконкурентной практике, поскольку была оштрафована за сговор на торгах в нескольких государственных тендерах, опубликованных правительственным агентством B, согласно статье, опубликованной 2022/01/10{tuple_delimiter}Согласно статье, опубликованной 2022/01/10, компания A была оштрафована за сговор на торгах при участии в нескольких государственных тендерах, опубликованных правительственным агентством B)
{record_delimiter}
(ЧЕЛОВЕК C{tuple_delimiter}НЕИЗВЕСТНО{tuple_delimiter}КОРРУПЦИЯ{tuple_delimiter}ПРЕДПОЛОЖИТЕЛЬНО{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Человек C подозревался в коррупционной деятельности в 2015 году{tuple_delimiter}Компания принадлежит человеку C, который подозревался в коррупционной деятельности в 2015 году)
{completion_delimiter}

-Real Data-
Используйте следующие входные данные для своего ответа.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output:"""


CONTINUE_PROMPT = "МНОГИЕ сущности были пропущены в последнем извлечении. Добавьте их ниже, используя тот же формат:\n"
LOOP_PROMPT = "Похоже, что некоторые объекты все еще могут быть пропущены. Ответьте ДА {tuple_delimiter} НЕТ, если все еще есть сущности, которые необходимо добавить.\n"
