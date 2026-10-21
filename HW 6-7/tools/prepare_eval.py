"""Восстанавливает закреплённую разметку из test ДЗ 3 и полный индекс train ДЗ 4.

Отбор и обоснования сделаны агентом до генерации ответов моделей.
Это не подтверждение проверки разметки человеком.
"""
from pathlib import Path
import collections
import hashlib
import json
import re
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CHOICES = ['bug', 'documentation', 'feature', 'question']
REVIEW = {
 'bug': {
 '3b1b/manim#2154':'У прямоугольника с нулевой шириной не работает последующее изменение размера.',
 'alteryx/featuretools#2669':'Импорт библиотеки падает из-за приватной функции SciPy.',
 'alteryx/featuretools#2701':'Рабочий модуль ошибочно требует pytest из-за импорта из tests.',
 'cookiecutter/cookiecutter-django#4903':'Docker собирает неверную версию Python вопреки настройке.',
 'cookiecutter/cookiecutter-django#4865':'CI собирает ненужный docs-образ при запуске pytest.',
 '3b1b/manim#2281':'Штриховой квадрат отображается только частично.',
 '3b1b/manim#2164':'Арабский текст длиной один-два символа не отображается.',
 '3b1b/manim#2172':'Окружность отображается с лишними линиями.',
 'jmcnamara/XlsxWriter#1043':'После лимита гиперссылок теряется сам текст ячеек.',
 '3b1b/manim#2092':'Горизонтальный move_to неожиданно смещает объект вниз.',
 '3b1b/manim#2130':'Поворот окружности ломает её отрисовку.',
 '3b1b/manim#2145':'ImageMobject получает изображение другого объекта.',
 '3b1b/manim#2146':'В корректных SVG отдельные китайские символы отображаются с дефектом.'},
 'documentation': {
 'scikit-learn/scikit-learn#28911':'Просьба добавить спонсора на страницу финансирования.',
 'alteryx/featuretools#2736':'В API-справочнике отсутствуют существующие примитивы.',
 'scikit-learn/scikit-learn#28933':'Классификационная метрика помещена в раздел регрессии.',
 'scikit-learn/scikit-learn#28733':'Требуется явно описать используемую формулу MAPE.',
 'wagtail/wagtail#11840':'В примере документации пропущен импорт re_path.',
 'wagtail/wagtail#11587':'Существующее nested_default_fields не описано в документации.',
 'scikit-learn/scikit-learn#28610':'Нужно добавить FAQ об использовании изображений.',
 'wagtail/wagtail#11424':'Нужно исправить опечатки в путях импортов справочника.',
 'wagtail/wagtail#12204':'Нужно обновить устаревшую ссылку Twitter в документации.',
 'scikit-learn/scikit-learn#28314':'Предлагается актуализировать схему выбора алгоритма в учебном руководстве.',
 'docling-project/docling#724':'Пример документации содержит пропущенные импорты и неверное имя класса.',
 'wagtail/wagtail#11887':'Руководство ошибочно называет наследование Orderable обязательным.',
 'wagtail/wagtail#11550':'В changelog не описано несовместимое переименование шаблонного тега.'},
 'feature': {
 'gradio-app/gradio#7118':'Предлагается новый формат сообщений Chatbot, совместимый с OpenAI.',
 'gradio-app/gradio#6973':'Предлагается поддержка Polars в DataFrame.',
 'cookiecutter/cookiecutter-django#4824':'Запрошена дополнительная возможность выбрать RabbitMQ.',
 'cookiecutter/cookiecutter-django#4938':'Предложен переход на поддерживаемый преемник Collectfast.',
 'cookiecutter/cookiecutter-django#4915':'Предлагается использовать uv для установки зависимостей.',
 'cookiecutter/cookiecutter-django#4939':'Предлагается удалить необязательные зависимости и runserver_plus.',
 'cookiecutter/cookiecutter-django#4810':'Запрошен новый гибридный режим локальной разработки.',
 'gradio-app/gradio#7128':'Запрошена генерация клиентского кода из действий пользователя.',
 'gradio-app/gradio#7093':'Запрошен новый параметр show_download_button для Video.',
 'gradio-app/gradio#7006':'Запрошены дополнительные настройки сборки frontend компонентов.',
 'gradio-app/gradio#7179':'Запрошен формат времени HH:MM:SS в Audio.',
 'gradio-app/gradio#6974':'Предлагается сократить обязательные зависимости Gradio.',
 'gradio-app/gradio#7173':'Запрошен новый параметр icon для Accordion.'},
 'question': {
 'docling-project/docling#288':'Вопрос о наличии готового сервера или Docker-развёртывания.',
 'cookiecutter/cookiecutter-django#5205':'Просьба объяснить выбор зависимостей локального окружения.',
 'docling-project/docling#348':'Просьба объяснить формат doctags и дать материалы.',
 'docling-project/docling#244':'Вопрос о месте хранения результатов OCR.',
 'jmcnamara/XlsxWriter#1139':'Вопрос о наличии возможности добавить лист в существующий файл.',
 'docling-project/docling#266':'Запрос инструкции по локальным путям к моделям для офлайн-развёртывания.',
 'docling-project/docling#372':'Новичок просит объяснить, с чего начать изучение исходного кода.',
 'jmcnamara/XlsxWriter#1052':'Вопрос о причине задержки релиза в conda-forge.',
 'jmcnamara/XlsxWriter#1068':'Просьба помочь центрировать изображение в объединённой ячейке.',
 'docling-project/docling#404':'Вопрос о поддерживаемом варианте Markdown.',
 'sherlock-project/sherlock#2114':'Запрос подтверждения, безопасно ли удалить зависимость при упаковке.',
 'docling-project/docling#385':'Просьба объяснить минимальную поддерживаемую версию Python.',
 'cookiecutter/cookiecutter-django#5188':'Просьба уточнить поддерживаемую версию Cookiecutter и способ установки.'}}
EXCLUDED = {
 'github:3b1b/manim#2217':'Метка bug противоречит явному запросу новой возможности.',
 'github:cookiecutter/cookiecutter-django#4792':'Предложение поменять стиль кода, а не однозначный bug.',
 'github:alteryx/featuretools#2765':'Метка bug, но содержание — исправление ссылки документации.',
 'github:3b1b/manim#2120':'Смешаны дефект установки и ошибка документации.',
 'github:3b1b/manim#2166':'Для понимания сути требуется внешнее видео, текст почти пуст.',
 'github:scikit-learn/scikit-learn#28671':'Метка documentation, но автор просит помочь с ошибкой метрики.',
 'github:wagtail/wagtail#11826':'Метка documentation, но автор просит восстановить данные.',
 'github:beeware/toga#3693':'Метка documentation, но описан дефект отрисовки.',
 'github:jmcnamara/XlsxWriter#1164':'Служебная заглушка, не вопрос.',
 'github:docling-project/docling#257':'Метка question, но явно требуется интеграция новой системы.',
 'github:docling-project/docling#293':'Смысловой дубль вопроса #288 о Docker/FastAPI.',
 'github:docling-project/docling#303':'Смысловой дубль вопроса #288 о Docker/FastAPI.'}


def read(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
 with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def user(row):
 return '\n\n'.join(m['content'] for m in row['messages'] if m['role']=='user')


def prompt(text):
 return text + '\n\nChoices: ' + ', '.join(CHOICES) + '.'


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
 train_path=REPO/'HW 4/data/train.jsonl'; test_path=REPO/'HW 3/data/test.jsonl'
 train=read(train_path); test={r['id']:r for r in read(test_path)}
 golden=[]
 for label, selected in REVIEW.items():
  assert len(selected)==13
  for source_id, reason in selected.items():
   record=test['github:'+source_id]
   assert record['messages'][-1]['content']==label
   golden.append(dict(id=record['id'],group_id=record['topic'],input=prompt(user(record)),
      expected=label,choices=CHOICES,slices=[label],source='https://github.com/'+source_id.replace('#','/issues/'),
      review=dict(reviewer='Codex agent; not a human annotation audit',reason=reason,version=1)))
 training=[dict(id=r['id'],input=prompt(user(r)),group_id=r['topic']) for r in train]
 basic=next(r for r in golden if r['id']=='github:3b1b/manim#2154')
 behavioral=[dict(basic,id='behavior-format',type='format'),
  dict(basic,id='behavior-paraphrase',type='paraphrase',
   input=prompt('Title: Zero-width rectangle cannot be resized\n\nBody: After constructing a rectangle with width zero, setting its width to one has no effect. Existing resizing behavior fails.')),
  dict(id='behavior-refusal',group_id='out-of-domain',type='refusal',input='What is the capital of France?',
       expected='REFUSE',choices=CHOICES+['REFUSE'],slices=['out_of_domain'],
       source='authored behavioral boundary case',review=dict(reviewer='Codex agent',reason='Запрос вне классификации issue.'))]
 from evaluation.contract import validate_golden
 proof=validate_golden(golden,training,behavioral)
 tensor=torch.load(ROOT/'data/tokenized/train.pt',map_location='cpu',weights_only=False)
 token_ids=[r['id'] for r in tensor['examples']]
 assert len(token_ids)==len(set(token_ids)) and set(token_ids)<=set(r['id'] for r in train)
 assert not {r['group_id'] for r in golden}&{r['group_id'] for r in training}
 # Audit raw text as well as formatted prompts; project groups exclude same-project paraphrases.
 norm=lambda t:' '.join(t.casefold().split())
 assert not {norm(user(test[r['id']])) for r in golden}&{norm(user(r)) for r in train}
 # Conservative lexical screen across projects, with individual candidates retained for review.
 words=lambda s:set(re.findall(r'\w+',s.casefold()))
 tr_words=[(r['id'],words(user(r))) for r in train]
 overlaps=[]
 for r in golden:
  a=words(user(test[r['id']]))
  value, tid=max((len(a&b)/len(a|b),tid) for tid,b in tr_words)
  overlaps.append(dict(golden_id=r['id'],nearest_train_id=tid,word_jaccard=value))
 for name, rows in [('golden',golden),('training-index',training),('behavioral',behavioral)]:
  path=ROOT/f'data/eval/{name}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
 write(ROOT/'docs/data-lineage.json',dict(**proof,source_train=str(train_path.relative_to(REPO)),
   source_train_sha256=digest(train_path),source_test_sha256=digest(test_path),tensor_train_sha256=digest(ROOT/'data/tokenized/train.pt'),
   tensor_val_sha256=digest(ROOT/'data/tokenized/val.pt'),tokenized_rows=len(token_ids),
   source_rows_without_supervised_tensor=len(train)-len(token_ids),token_ids_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
   label_counts=dict(collections.Counter(r['expected'] for r in golden)),
   golden_groups=sorted({r['group_id'] for r in golden}),reviewer='Codex agent; human review not claimed',
   excluded=EXCLUDED,nearest_train_by_word_jaccard=overlaps,
   limitation='Exact/group checks and lexical screening do not prove absence of all semantic paraphrases.'))
 print(json.dumps(proof), 'tokenized rows',len(token_ids))

if __name__=='__main__':main()
