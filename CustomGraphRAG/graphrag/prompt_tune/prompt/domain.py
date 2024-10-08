# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for domain generation."""

GENERATE_DOMAIN_PROMPT = """
Ты - умный помощник, который помогает человеку анализировать информацию, содержащуюся в текстовом документе.
Предоставив образец текста, помоги пользователю, указав область описания, в которой кратко излагается суть текста.
Примерами таких областей являются: "Социальные исследования", "Алгоритмический анализ", "Медицинские науки" и другие.

Текст: {input_text}
Домен:"""
