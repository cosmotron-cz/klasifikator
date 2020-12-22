# Klasifikátor

Nástroj pre vecnú klasifikáciu dokumentov, na základe metadát a textov.

# Inštalácia

Návod je pre Windows, ale podobný postup je aj pre Linux.

1. Inštalácia python 3.7 - [link](https://www.python.org/downloads/)
2. Spustiť cmd
3. Inštalácia requirements.txt - `pip install -r requirements.txt`
4. Stiahnuť Elasticsearch 7.2 - [link](https://www.elastic.co/downloads/past-releases/elasticsearch-7-2-0)
5. Rozbaliť stiahnutý .zip
6. Do súboru elasticsearch-7.2.0\config\elasticsearch.yml pridať riadok *http.max_content_length: 500mb* na koniec
7. V súbore elasticsearch-7.2.0\config\jvm.options zmeniť riadok *-Xms1g* na *-Xms4g*
8. V súbore elasticsearch-7.2.0\config\jvm.options zmeniť riadok *-Xmx1g* na *-Xmx4g*
9. Stiahnuť tento repozitár
10. Rozbaliť súbory models\fulltext\fulltext.zip a models\keywords\keywords1 a models\keywords\keywords2 do priečinkov v ktorom sa nachádzajú


# Použitie

Pre spustenie klasifikácie je potrebné najpr spustiť elastic - elasticsearch-7.2.0\bin\elasticsearch a potom spusiť skript subject_classifier.py:
`python subject_classifier.py --directory [data] --export_to [xml]`

Priečinok data by mal obsahovať tieto súbory/adresáre:

- Priečinok text - obsahuje všetky texty k súborom s názvom uuid_\[uuid\].tar.gz, napríklad uuid_1e520020-10ab-11e4-8e0d-005056827e51.tar.gz
- Priečinok sorted_pages - obsahuje zoradenie strán textov s názvom uuid_\[uuid\].txt napríklad uuid_1e520020-10ab-11e4-8e0d-005056827e51.txt
- Súbor sloucena_id - obsahuje páry OAI a uuid
- Súbor metadata.xml - obsahuje metadata pre dokumenty vo formáte MARC XML

Je možné použiť nepovinný parameter --action ACTION kde ACTION može byť:

- import - spustí import dát
- classify - spustí klasifikáciu dát
- export - exportuje dáta z elastiku
- all - vykonajú sa akcie import, classify a export
- remove - vymaže dáta z elastiku

Pri akcii export sa dáta uložia do zadaného xml. Pre každý záznam zo vstupného xml(metadata.xml) sa pridajú polia N072 pre generovaný konspket a N650 pre generované klúčové slová.

# Použitie API
Pre spustenie servera na ktorom beží API pre klasifikáciu je potrebné:
1. Spustiť elastic - elasticsearch-7.2.0\bin\elasticsearch
2. Spustiť skript python_api.py
3. V dalšom procese spustiť runner.py

