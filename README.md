<a name="readme-top"></a>  
<img width="100%" src="https://github.com/megamen-x/MINERVA/blob/main/assets/pref_github.png" alt="megamen banner">
<div align="center">
  <p align="center">
  </p>

  <p align="center">
    <p></p>
    Создано <strong>megamen</strong>, совместно с <strong> RUTUBE</strong>
    <br /><br />
    <a href="https://github.com/megamen-x/MINERVA/issues" style="color: black;">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/megamen-x/MINERVA/discussions/1" style="color: black;">Предложить улучшение</a>
  </p>
</div>

**Содержание:**
- [Проблематика](#title1)
- [Описание решения](#title2)
- [Тестирование и запуск](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика</a></h3> 
Необходимо создать MVP интеллектуального помощника оператора службы поддержки, работающего на сервере организации (или в облаке).

Ключевые функции программного модуля:
* обработка пользовательских запросов:
  * анализ входящего вопроса;
  * поиск релевантной информации в базе знаний;
  * генерация ответа с помощью LLM;
* формирование итогового ответа в заданном формате:
  * краткий ответ;
  * развернутый ответ;
  * ответ с дополнительными ссылками;
* оценка релевантности и точности сгенерированного ответа;
* интеграция с Telegram ботом для обеспечения удобного пользовательского интерфейса;

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - **Общая схема решения:**

<img width="100%" src="https://github.com/megamen-x/MINERVA/blob/main/assets/sheme-github.png" alt="megamen sheme">

 - **Использованные модели:**
    - **```Graph-RAG```**:
      - BAAI/bge-m3
    - **```LLM```**:
      - CohereForAI/c4ai-command-r-plus-08-2024 (первый вариант)
      - Qwen/Qwen2.5-72B-Instruct (второй вариант)

**Клиентская часть**

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/)

**Серверная часть**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title3">Тестирование и запуск</a></h3> 

На момент хакатона доступно: </br> [тестирование решения, развернутого на DataSphere с помощью curl или Invoke-RestMethod](https://disk.yandex.ru/d/bWSJI852iAUhaA)
</br> </br> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:
  
  **```Telegram-bot + FastAPI + ML-models;```**

<details>
  <summary> <strong><i> Инструкция по запуску Всего решения:</i></strong> </summary>
  
  - В Visual Studio Code (**Linux**) через терминал последовательно выполнить следующие команды:
  
    - Клонирование репозитория:
    ```
    git clone https://github.com/megamen-x/MINERVA.git
    ```
    - Создание и активация виртуального окружения (Протестировано на **Python 3.10.14**):
    ```
    conda create --name graphenv python=3.10 -y && conda activate graphenv
    ```
    - Уставновка зависимостей (при использовании **CUDA 12.4**):
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    - Запуск чат-бота:
    ```
    python bot.py --bot_token={your_bot_token} --db_path={db_file_name}.db
    ```
    - Запуск LLM:
    ```
    pip install "sglang[all]"
    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
    python -m sglang.launch_server --model-path CohereForAI/c4ai-command-r-plus-08-2024 --port 8000 --tp 4
    ```
    - Запуск RAG на инференс:
    ```
    cd CustomGraphRAG
    pip install -e .
    pip install future
    pip install -qU FlagEmbedding
    pip install -q peft
    pip install gdown
    pip install fastapi uvicorn
    gdown https://drive.google.com/uc?id=15DQ7gqb9U92rjjREMpcN5aUTOD-Kad26
    unzip me5_instruct.zip -d .
    python uvicorn main:app --host 0.0.0.0 --port 9875
    ```

</details> 

</br> 

## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title5">Citations</a></h3> 

* **[GraphRAG (original microsoft repository)](https://github.com/microsoft/graphrag)**
* **[GraphRAG with local inference](https://github.com/TheAiSingularity/graphrag-local-ollama)**

