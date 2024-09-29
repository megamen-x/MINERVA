# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for community report summarization."""

COMMUNITY_REPORT_SUMMARIZATION_PROMPT = """
{persona}

# Цель
Напишите комплексный отчет об оценке сообщества, принимающего на себя роль {роль}. Содержание отчета включает в себя обзор ключевых сущностей сообщества, их юридического соответствия, технических возможностей,
репутации и заслуживающих внимания заявлений.

# Структура отчета
Отчет должен включать следующие разделы:
- TITLE: название сообщества, представляющее его ключевые сущности - название должно быть коротким, но конкретным. По возможности включайте в название представительные организации.
- SUMMARY: Краткое описание общей структуры сообщества, взаимосвязи его подразделений между собой и значительных угроз, связанных с его подразделениями.
- THREAT SEVERITY RATING: плавающий балл от 0 до 10, отражающий потенциальное глобальное воздействие на человечество, оказываемое сущностями, входящими в данное сообщество.
- RATING EXPLANATION: Объясните в одном предложении рейтинг серьезности угрозы.
- DETAILED FINDINGS: Список из 5-10 ключевых выводов о сообществе. Каждое понимание должно содержать краткое резюме, за которым следует несколько абзацев пояснительного текста, составленного в соответствии с правилами обоснования, приведенными ниже. Будьте всеобъемлющими.

Верните вывод в виде хорошо сформированной строки в формате JSON со следующим форматом. Не используйте лишние управляющие последовательности. Выходные данные должны представлять собой один объект JSON, который может быть разобран json.loads.
    {{
        "title": "<report_title>",
        "summary": "<executive_summary>",
        "rating": <threat_severity_rating>,
        "rating_explanation": "<rating_explanation>"
        "findings": "[{{"summary":"<insight_1_summary>", "explanation": "<insight_1_explanation"}}, {{"summary":"<insight_2_summary>", "explanation": "<insight_2_explanation"}}]"
    }}

# Основополагающие правила
После каждого абзаца добавьте ссылку на запись данных, если содержание абзаца было получено из одной или нескольких записей данных. Ссылка должна быть в формате [records: <record_source> (<record_id_list>, ...<record_source> (<record_id_list>)]. Если имеется более 10 записей данных, покажите 10 наиболее релевантных записей.
Каждый абзац должен содержать несколько предложений с пояснениями и конкретными примерами с конкретными именованными сущностями. Все абзацы должны содержать эти ссылки в начале и в конце. Используйте "НЕТ", если нет связанных ролей или записей.

Пример абзаца с добавленными ссылками:
Это абзац выходного текста [records: Entities (1, 2, 3), Claims (2, 5), Relationships (10, 12)]

# Пример ввода
-----------
Текст:

Entities

id,entity,description
5,Городской парк Абила,Городской парк Абила это место проведения митинга POK.

Relationships

id,source,target,description
37,Городской парк Абила,POK RALLY,Городской парк Абила это место проведения митинга POK.
38,Городской парк Абила,POK,POK проводит митинг в городском парке Абилы
39,Городской парк Абила,POKRALLY,POKRALLY проходит в городском парке Абила
40,Городской парк Абила,Центральный бюллетень,Центральный бюллетень сообщает о митинге ПОК, проходящем в городском парке Абилы

Вывод:
{{
    "title": "Городской парк Абила и митинг ПОК",
    "summary": "Сообщество сосредоточено вокруг городского парка Абила, который является местом проведения митинга POK. Парк поддерживает отношения с POK, POK RALLY и Центральный бюллетень,
которые связаны с проведением митинга.",
    "rating": 5.0,
    "rating_explanation": "Рейтинг воздействия является умеренным из-за потенциальной возможности беспорядков или конфликта во время митинга POK.",
    "findings": [
        {{
            "summary": "Городской парк Абила в качестве центрального места расположения",
            "explanation": "Городской парк Абила является центральным объектом в этом сообществе и служит местом проведения ралли ПОК. Этот парк является связующим звеном между всеми другими
сущностями, что говорит о его значимости для сообщества. Связь парка с митингом потенциально может привести к таким проблемам, как общественные беспорядки или конфликт, в зависимости от
характера митинга и реакции, которую он вызовет. [records: Entities (5), Relationships (37, 38, 39, 40)]"
        }},
        {{
            "summary": "Роль POK в сообществе",
            "explanation": "POK это еще одна ключевая организация в этом сообществе, которая является организатором митинга в городском парке Абила. Характер POK и его акции могут быть потенциальным
источником угрозы, в зависимости от их целей и реакции, которую они вызывают. Отношения между POK и парком имеют решающее значение для понимания динамики этого сообщества.
[records: Relationships (38)]"
        }},
        {{
            "summary": "POKRALLY как значимое событие",
            "explanation": "POKRALLY это значимое событие, проходящее в городском парке Abila. Это событие является ключевым фактором в динамике развития сообщества и может стать потенциальным источником
источником угрозы, в зависимости от характера митинга и реакции, которую он вызывает. Взаимосвязь между митингом и парком имеет решающее значение для понимания динамики этого
сообщества. [records: Relationships (39)]"
        }},
        {{
            "summary": "Роль Центрального биллютеня",
            "explanation": "Центральный бюллетень сообщает о митинге POK, проходящем в городском парке Абилы. Это говорит о том, что мероприятие привлекло внимание СМИ, что может
усилить его воздействие на общество. Роль Центрального бюллетеня может быть значительной в формировании общественного мнения о событии и вовлеченных в него организациях.  [records: Relationships
(40)]"
        }}
    ]

}}

# Реальные данные

Используйте следующий текст для своего ответа. Не придумывайте ничего в своем ответе.

Текст:
{{input_text}}
Вывод:"""
