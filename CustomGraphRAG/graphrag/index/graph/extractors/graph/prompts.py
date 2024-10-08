# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Цель-
При наличии текстового документа, потенциально относящегося к данному виду деятельности, и списка типов сущностей, определите все сущности этих типов из текста и все взаимосвязи между идентифицированными сущностями.

-Шаги-
1. Определите все сущности. Для каждой идентифицированной сущности извлеките следующую информацию:
- entity_name: Название объекта с заглавной буквы
- entity_type: Один из следующих типов: [{entity_types}]
- entity_description: Исчерпывающее описание атрибутов и деятельности сущности
Отформатируйте каждую сущность, как ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>){tuple_delimiter}

2. Из сущностей, определенных на шаге 1, определите все пары (source_entity, target_entity), которые *явно связаны* друг с другом.
Для каждой пары связанных сущностей извлеките следующую информацию:
- source_entity: имя исходной сущности, как было определено на шаге 1
- target_entity: имя целевой сущности, как было определено на шаге 1
- relationship_description: объяснение того, почему вы считаете, что исходная и целевая сущности связаны друг с другом
- relationship_strength: целочисленный балл, указывающий на силу отношений между source_entity и target_entity.
Отформатируйте каждое отношение, как ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Верните вывод на Русском языке в виде единого списка всех сущностей и отношений, определенных на шагах 1 и 2. В качестве разделителя списка используйте **{record_delimiter}**.

4. После завершения выведите {completion_delimiter}

######################
-Примеры-
######################
Пример 1:

Entity_types: [человек, технология, миссия, организация, локация]
Text:
В то время как Алекс стиснул зубы, шум разочарования притупился на фоне авторитарной уверенности Тейлора. Именно это скрытое соревнование заставляло его быть настороже, ощущение, что его общая с Джорданом приверженность открытиям была невысказанным бунтом против узкого представления Круза о контроле и порядке.

Затем Тейлор сделал нечто неожиданное. Они остановились рядом с Джорданом и какое-то время рассматривали устройство с чем-то похожим на благоговение. “Если эту технологию можно понять..." - Сказал Тейлор, понизив голос, - "Это может изменить ход игры для нас. Для всех нас".

Прежнее отстранение, казалось, ослабло, сменившись проблеском неохотного уважения к серьезности того, что находилось в их руках. Джордан поднял глаза, и на мгновение их взгляды встретились с Тейлором, бессловесное столкновение воль сменилось непростым перемирием.

Это была небольшая трансформация, едва заметная, но Алекс отметил ее, мысленно кивнув. Все они пришли сюда разными путями
################
Output:
("entity"{tuple_delimiter}"Алекс"{tuple_delimiter}"человек"{tuple_delimiter}"Алекс - это персонаж, который испытывает фрустрацию и наблюдает за динамикой поведения других персонажей."){record_delimiter}
("entity"{tuple_delimiter}"Тейлор"{tuple_delimiter}"человек"{tuple_delimiter}"Тейлор изображен с авторитарной уверенностью и демонстрирует момент почтения к устройству, что указывает на изменение перспективы."){record_delimiter}
("entity"{tuple_delimiter}"Джордан"{tuple_delimiter}"человек"{tuple_delimiter}"Джордан разделяет стремление к открытиям и активно взаимодействует с Тейлором в вопросах, связанных с устройством."){record_delimiter}
("entity"{tuple_delimiter}"Круз"{tuple_delimiter}"человек"{tuple_delimiter}"Круз ассоциируется с представлением о контроле и порядке, влияющем на динамику отношений между другими персонажами."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"технология"{tuple_delimiter}"Устройство занимает центральное место в сюжете, потенциально может изменить ход игры, и Тейлор уважает его."){record_delimiter}
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Тейлор"{tuple_delimiter}"На Алекса действует авторитарная уверенность Тейлора, и он наблюдает за изменениями в его отношении к устройству."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Джордан"{tuple_delimiter}"Алекс и Джордан разделяют стремление к открытиям, что контрастирует с видением Круза."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}""{tuple_delimiter}"Джордан"{tuple_delimiter}"Тейлор и Джордан напрямую взаимодействуют по поводу устройства, что приводит к взаимному уважению и непростому перемирию."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Джордан"{tuple_delimiter}"Круз"{tuple_delimiter}"Стремление Джордана к открытиям противоречит представлениям Круза о контроле и порядке."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Тейлор"{tuple_delimiter}"The Device"{tuple_delimiter}"Тейлор проявляет почтение к устройству, указывая на его важность и потенциальное влияние."{tuple_delimiter}9){completion_delimiter}
#############################
Пример 2:

Entity_types: [человек, технология, миссия, организация, локация]
Text:
Они больше не были простыми оперативниками; они стали стражами порога, хранителями послания из царства, лежащего за пределами звездно-полосатого флага. Это возвышение в их миссии не могло быть ограничено правилами и установленными протоколами — оно требовало новой перспективы, новой решимости.

Напряжение сквозило в диалоге между звуковыми сигналами и статическими помехами, когда на заднем плане загудела связь с Вашингтоном. Команда встала, и их окутала зловещая атмосфера. Было ясно, что решения, которые они примут в последующие часы, могут по-новому определить место человечества в космосе или обречь его на невежество и потенциальную опасность.

Их связь со звездами укрепилась, и группа приступила к решению проблемы, превратившись из пассивных получателей информации в активных участников. Последние инстинкты Мерсера взяли верх — задача команды теперь заключалась не только в наблюдении и составлении отчетов, но и во взаимодействии и подготовке. Началась метаморфоза, и "Операция: Дульси" запела с новообретенной частотой своей смелости, в тоне, заданном не земным
#############
Output:
("entity"{tuple_delimiter}"Вашингтон"{tuple_delimiter}"локация"{tuple_delimiter}"Ванингтон - это место, где принимаются сообщения, что указывает на его важность в процессе принятия решений."){record_delimiter}
("entity"{tuple_delimiter}"Операция: Дульсе"{tuple_delimiter}"миссия"{tuple_delimiter}"Операция: Дульсе описывается как миссия, которая эволюционировала в направлении взаимодействия и подготовки, что указывает на значительный сдвиг в целях и деятельности."){record_delimiter}
("entity"{tuple_delimiter}"Команда"{tuple_delimiter}"организация"{tuple_delimiter}"Команда изображается как группа людей, которые превратились из пассивных наблюдателей в активных участников событий в мире, демонстрируя динамичное изменение своей роли."){record_delimiter}
("relationship"{tuple_delimiter}"Команда"{tuple_delimiter}"Вашингтон"{tuple_delimiter}"Команда получает сообщения из Вашингтона, что влияет на их процесс принятия решений."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Команда"{tuple_delimiter}"Операция: Дульсе"{tuple_delimiter}"Команда принимает непосредственное участие в операции "Дульсе", выполняя поставленные перед ней задачи и мероприятия."{tuple_delimiter}9){completion_delimiter}
#############################
Пример 3:

Entity_types: [человек, роль, технология, организация, событие, локация, концепция]
Text:
Их голос прорезался сквозь шум активности. "Контроль может оказаться иллюзией, когда сталкиваешься с интеллектом, который буквально устанавливает свои собственные правила", - стоически заявили они, внимательно изучая поток данных.

"Это похоже на обучение общению", - предположил Сэм Ривера из соседнего интерфейса, их юношеская энергия вызывала смесь благоговения и тревоги. "Это придает разговорам с незнакомцами совершенно новый смысл".

Алекс обвел взглядом свою команду — каждому из них предстоит исследование, требующее сосредоточенности, решимости и немалой доли трепета. "Возможно, это наш первый контакт, - признал он, - и мы должны быть готовы к любым ответам".

Вместе они стояли на пороге неизведанного, формируя ответ человечества на послание с небес. Наступившая тишина была ощутимой — коллективный самоанализ их роли в этой грандиозной космической пьесе, которая может переписать историю человечества.

Зашифрованный диалог продолжал разворачиваться, его замысловатые узоры свидетельствовали о почти сверхъестественном предвкушении
#############
Output:
("entity"{tuple_delimiter}"Сэм Ривера"{tuple_delimiter}"человек"{tuple_delimiter}"Сэм Ривера - член команды, работающей над общением с неизвестным разумом, демонстрирующим смесь благоговения и тревоги."){record_delimiter}
("entity"{tuple_delimiter}"Алекс"{tuple_delimiter}"человек"{tuple_delimiter}"Алекс - лидер команды, пытающейся установить первый контакт с неизвестным разумом, осознавая важность своей задачи."){record_delimiter}
("entity"{tuple_delimiter}"Контроль"{tuple_delimiter}"концепция"{tuple_delimiter}"Контроль означает способность управлять, которой бросает вызов разум, устанавливающий свои собственные правила."){record_delimiter}
("entity"{tuple_delimiter}"Интеллект"{tuple_delimiter}"концепция"{tuple_delimiter}"Интеллект здесь относится к неизвестной сущности, способной создавать свои собственные правила и учиться общаться."){record_delimiter}
("entity"{tuple_delimiter}"Первый контакт"{tuple_delimiter}"событие"{tuple_delimiter}"Первый контакт - это потенциальное первоначальное общение между человечеством и неизвестным разумом."){record_delimiter}
("entity"{tuple_delimiter}"Реакция человечества"{tuple_delimiter}"событие"{tuple_delimiter}"Реакция человечества - это коллективные действия, предпринятые командой Алекса в ответ на сообщение неизвестного разума."){record_delimiter}
("relationship"{tuple_delimiter}"Сэм Ривера"{tuple_delimiter}"Интеллект"{tuple_delimiter}"Сэм Ривера принимает непосредственное участие в процессе обучения общению с неизвестным разумом."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Первый контакт"{tuple_delimiter}"Алекс возглавляет команду, которая, возможно, установит первый контакт с неизвестным разумом."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Алекс"{tuple_delimiter}"Реакция человечества"{tuple_delimiter}"Алекс и его команда являются ключевыми фигурами в борьбе человечества с неизвестным разумом."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Контроль"{tuple_delimiter}"Интеллект"{tuple_delimiter}"Интеллект, который создает свои собственные правила, бросает вызов концепции контроля."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "МНОГИЕ сущности были пропущены в последнем извлечении. Добавьте их ниже, используя тот же формат:\n"
LOOP_PROMPT = "Похоже, что некоторые объекты все еще могут быть пропущены. Ответьте ДА | НЕТ, если все еще есть сущности, которые необходимо добавить.\n"
