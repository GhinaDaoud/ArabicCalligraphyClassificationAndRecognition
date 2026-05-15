# Zero-shot Arabic calligraphy OCR — summary

_Eval set: 49 stratified samples (7 per style) from HICMA + DuwaBench, with ground-truth Arabic transcripts. Arabic normalization (alif/yaa/taa-marbuta unification, tashkeel + kashida stripped) applied before scoring._


# Headline comparison (overall)

| model | CER | WER | BLEU |
|---|---|---|---|
| ain-7b-lora-ep1 | 0.569 | 0.838 | 36.06 |
| ain-7b-lora-ep2 | 0.470 | 0.746 | 36.75 |
| ain-7b-lora | 0.517 | 0.801 | 39.21 |
| ain-7b-routed | 0.353 | 0.603 | 41.10 |
| ain-7b | 0.492 | 0.795 | 22.64 |
| baseer-nakba | 0.559 | 0.806 | 14.56 |
| ketaba-lora | 0.552 | 0.887 | 12.51 |
| qari-v0.2.2.1 | 0.782 | 1.163 | 6.92 |
| qari-v0.3 | 2.126 | 3.021 | 5.66 |
| sherif-handwritten-v3 | 0.676 | 0.954 | 15.87 |

## ain-7b-lora-ep1

_Source: `calligraphy_ocr/results/zero_shot/ain-7b-lora-ep1_predictions.csv` (49 rows)_

- mean latency: **0.54s**, p95: **1.09s** per image
- peak VRAM observed: **16592 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.422 | 0.707 | 20.90 |
| Kufic | 1.484 | 1.571 | 2.28 |
| Muhaqaq | 0.028 | 0.071 | 85.82 |
| Naskh | 0.061 | 0.260 | 59.56 |
| Nasta'liq | 0.778 | 1.226 | 5.22 |
| Ruq'ah | 0.867 | 1.419 | 29.38 |
| Thuluth | 0.340 | 0.607 | 49.28 |
| **overall** | 0.569 | 0.838 | 36.06 |


#### Qualitative — ain-7b-lora-ep1


##### Diwani

**3 best (lowest CER):**

- `1643-Diwani1_segment_no_0.jpg` — CER 0.171
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئلون أهل الذكر إن كنتم تعلمون
- `dwn71.jpg` — CER 0.182
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: ويتولا الله بينما تظنّ أنك ممْضى
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.200
  - GT:   ولذكر الله أكبر
  - Pred: ولنذر الله الكبر

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.875
  - GT:   له الملك
  - Pred: الرحمن
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.660
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: ولله الأسماء الحسنى والمعوذة والله على كل شيء عليما
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.634
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: والدين لا تؤذى أهل الأرض من بعد موتها ولا يؤذيها

##### Kufic

**3 best (lowest CER):**

- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.333
  - GT:   الودود
  - Pred: الود
- `kuf14.jpg` — CER 0.824
  - GT:   الصبر مفتاح الفرج
  - Pred: الله
- `kuf84.jpg` — CER 1.000
  - GT:   لا تحزن
  - Pred: نعمنا

**3 worst (highest CER):**

- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 4.500
  - GT:   الله
  - Pred: بسم الله الرحمن الرحيم
- `kuf19.jpg` — CER 1.400
  - GT:   نحلم ونحقق
  - Pred: يا الله إني لكوفي
- `kuf66.jpg` — CER 1.250
  - GT:   اقرأ
  - Pred: لا تقربوا

##### Muhaqaq

**3 best (lowest CER):**

- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.000
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قل الله ثم ذرهم في خوضهم يلعبون
- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم

**3 worst (highest CER):**

- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.154
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين أهدينا الصراط المستقيم
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.040
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العقل ومن شر حاسد اذا حسد
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العالمين الرحمن الرحيم

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.000
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفلق من شر ما خلق ومن شر
- `442 -1.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العالمين الرحمن الرحيم
- `778-3.jpg` — CER 0.037
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فسيحشرهم إليه جميعا فأما الذين أمنوا وعملوا الصالحات

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.146
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أعلنت لهم وأشرت لهم أشرا إن فقلك استغفروا
- `1001-1.jpg` — CER 0.130
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: ويغفو اعز كثير قل جآءكم من الله نور و كتاب مبين
- `890-1.jpg` — CER 0.074
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: للمتقين عند ربهم جنات النعيم افتحعل المسلم كالمجرمين

##### Nasta'liq

**3 best (lowest CER):**

- `nas19.jpg` — CER 0.269
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: البين وجدت مسلق وكلام لين
- `nas32.jpg` — CER 0.407
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: و على الديب شر كلّ المتشوكون
- `nas47.jpg` — CER 0.667
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: لا تحسب المُحَدِّمَ أَنْتَ اكْلُهُ لَنْ يَسْلَعَ المجدَ حَتَّى تَلْقَى لصباً سيراً

**3 worst (highest CER):**

- `nas49.jpg` — CER 1.407
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: اعلم ما تشاء من علم اعجاز الله اذا علم مسك صدق الله العظيم
- `6_376.jpg` — CER 1.091
  - GT:   فباي الاء ربكما تكذبان
  - Pred: فَأَيّ رُحْمَةٍ مِنْهَا تَكْذِبُونَ الَّتِي كُلِّها فَرْعٌ
- `nas51.jpg` — CER 0.818
  - GT:   نور علي نور
  - Pred: الله أكبر

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.000
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عليه توكّلت وإليه أنيب
- `ruq53.jpg` — CER 0.185
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: فلس يتغيّر العالم بحزنك ألم
- `ruq56.jpg` — CER 0.467
  - GT:   يارب أمور ميسرة
  - Pred: يا رب أمني رحمة

**3 worst (highest CER):**

- `ruq26.jpg` — CER 3.167
  - GT:   يهلك المرء بأفكاره
  - Pred: يا جهلك أن تعلم إن الظالمات لا يخلفن أجرهم في جهنم إلا يحبسها ربه لترضى
- `ruq49.jpg` — CER 1.000
  - GT:   انت عمري
  - Pred: استغفر الله
- `ruq4.jpg` — CER 0.639
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط الهندسة روحانية ظهرت بآلة جسمانية

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سُبحان الله
- `Allah-1-Black_segment_no_0.jpg` — CER 0.000
  - GT:   الله
  - Pred: الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.059
  - GT:   لكم دينكم ولي دين
  - Pred: لكم دينكم ولي ديني

**3 worst (highest CER):**

- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 1.037
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: وكان في قلوبهم آناء من ينطق بالحق وهو موقوت
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 0.850
  - GT:   كل شيء هالك الا وجهه
  - Pred: كل شيء له علامة إلا أن تمحى عليه
- `2_108.jpg` — CER 0.278
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله سُبحان الله وتعالى

## ain-7b-lora-ep2

_Source: `calligraphy_ocr/results/zero_shot/ain-7b-lora-ep2_predictions.csv` (49 rows)_

- mean latency: **0.53s**, p95: **1.02s** per image
- peak VRAM observed: **16592 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.421 | 0.702 | 20.17 |
| Kufic | 0.984 | 1.167 | 15.00 |
| Muhaqaq | 0.078 | 0.167 | 77.95 |
| Naskh | 0.094 | 0.367 | 47.42 |
| Nasta'liq | 0.768 | 1.310 | 5.51 |
| Ruq'ah | 0.359 | 0.764 | 42.18 |
| Thuluth | 0.589 | 0.748 | 49.05 |
| **overall** | 0.470 | 0.746 | 36.75 |


#### Qualitative — ain-7b-lora-ep2


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.152
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: ويتولى الله بينما تظنّ أنك مفرورٌ
- `1643-Diwani1_segment_no_0.jpg` — CER 0.171
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئلون أهل الذكر إن كنتم تعلمون
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.200
  - GT:   ولذكر الله أكبر
  - Pred: ونذر الله البر

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.875
  - GT:   له الملك
  - Pred: الرحمن
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.805
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: والسلام من ربه توفيقي اني لا أزيد الا بذكر الله
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.468
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: ويستهنو الرسول والفضل وكان عليما حكما

##### Kufic

**3 best (lowest CER):**

- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.333
  - GT:   الودود
  - Pred: الود
- `kuf14.jpg` — CER 0.706
  - GT:   الصبر مفتاح الفرج
  - Pred: الله جل جلاله
- `2_153.jpg` — CER 1.000
  - GT:   وذو حس فكاهي
  - Pred: فكلمة وذو حس

**3 worst (highest CER):**

- `kuf66.jpg` — CER 1.500
  - GT:   اقرأ
  - Pred: لقد أقرضت
- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 1.250
  - GT:   الله
  - Pred: الله أكبر
- `kuf19.jpg` — CER 1.100
  - GT:   نحلم ونحقق
  - Pred: يا الله انيكفي

##### Muhaqaq

**3 best (lowest CER):**

- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.000
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قل الله ثم ذرهم في خوضهم يلعبون
- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم

**3 worst (highest CER):**

- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.359
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين أهدنا الصراط
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.160
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العقل ومن شر حاسد إن أحسد
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.029
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمدلله رب العالمين الرحمن الرحيم

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.000
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفلق من شر ما خلق ومن شر
- `442 -1.jpg` — CER 0.029
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمدلله رب العالمين الرحمن الرحيم
- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 0.037
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: سبحان ربك رب العزة عما يصفون 0

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.268
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: اعلنت لهم و اشرفت لهم أشرا رافقك استغفروا في
- `890-1.jpg` — CER 0.167
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: ليمتقنوا عند ربهم جنان النعيم افتحعل المسلمين كالجرميين
- `778-3.jpg` — CER 0.093
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فسبع حشرهم إليه جميعا فأما الذين أمنوا وعملوا الصالحات

##### Nasta'liq

**3 best (lowest CER):**

- `nas47.jpg` — CER 0.246
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: لن يبلغ المجد حتى لعق له صبر لا تحسب المجدم أن تشك كله
- `nas32.jpg` — CER 0.259
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وعلى الدي شر وكل المتواكلون
- `nas19.jpg` — CER 0.654
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: وَكَلامٍ لِيْئِنْ وَجَدٌّ طُلقاتٌ هَسْتَينِ البَرِّينِ

**3 worst (highest CER):**

- `nas35.jpg` — CER 1.158
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: المرء مع من أحب و يبغض من شاء
- `6_376.jpg` — CER 1.136
  - GT:   فباي الاء ربكما تكذبان
  - Pred: فَمَايَكُونَ مِنْ أَحَدٍ إِلَّا رَبُّكُمْ هُوَ الْأَعْزُ بِرَاءِهِمْ
- `nas51.jpg` — CER 1.000
  - GT:   نور علي نور
  - Pred: لم يلد ولم يولد

##### Ruq'ah

**3 best (lowest CER):**

- `ruq4.jpg` — CER 0.000
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: ظهرت بآلة جسمانية الخط هندسة روحانية
- `ruq1.jpg` — CER 0.000
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عليه توكلت وإليه أنيب
- `ruq53.jpg` — CER 0.185
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: فلس يتغير العالم بحزنك ألم

**3 worst (highest CER):**

- `ruq49.jpg` — CER 0.750
  - GT:   انت عمري
  - Pred: انت عمي جاليك
- `ruq26.jpg` — CER 0.722
  - GT:   يهلك المرء بأفكاره
  - Pred: يا جهلك على المرء أفق طاع
- `ruq56.jpg` — CER 0.467
  - GT:   يارب أمور ميسرة
  - Pred: يا رب اامورك سجرا

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سبحان الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.000
  - GT:   لكم دينكم ولي دين
  - Pred: لكم دينكم ولي دين
- `2_108.jpg` — CER 0.056
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله سبحانه وتعالى

**3 worst (highest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 2.250
  - GT:   الله
  - Pred: الله جل جلاله
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 0.889
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: و كان في قلوبهم آثار نصر الله الموفى بها
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 0.500
  - GT:   كل شيء هالك الا وجهه
  - Pred: كل شيء علما إلا اللہ وحده

## ain-7b-lora

_Source: `calligraphy_ocr/results/zero_shot/ain-7b-lora_predictions.csv` (49 rows)_

- mean latency: **0.58s**, p95: **1.16s** per image
- peak VRAM observed: **16592 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.510 | 0.832 | 21.08 |
| Kufic | 0.770 | 1.000 | 21.43 |
| Muhaqaq | 0.061 | 0.119 | 84.48 |
| Naskh | 0.084 | 0.309 | 59.93 |
| Nasta'liq | 1.142 | 1.679 | 8.53 |
| Ruq'ah | 0.309 | 0.721 | 40.40 |
| Thuluth | 0.743 | 0.948 | 38.60 |
| **overall** | 0.517 | 0.801 | 39.21 |


#### Qualitative — ain-7b-lora


##### Diwani

**3 best (lowest CER):**

- `1643-Diwani1_segment_no_0.jpg` — CER 0.086
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئلون أهل الذكر إن كنتم لا تعلمون
- `dwn71.jpg` — CER 0.121
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: ويتولى الله بينما تظن أنك بمفرور
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.221
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: فاطر المحمودات والأرض أن تنبئي في الدنيا وآخرة توفى مسماً وطغيى بالصالحين

**3 worst (highest CER):**

- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 1.085
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: ولله الأسماء الحسنى فادعوه بها فلن يرضى لك ما عملت وما سواك ولا تدعو به شيئا
- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.875
  - GT:   له الملك
  - Pred: الرحمن
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.600
  - GT:   ولذكر الله أكبر
  - Pred: والنظر السيئة للهبر

##### Kufic

**3 best (lowest CER):**

- `kuf66.jpg` — CER 0.000
  - GT:   اقرأ
  - Pred: إقرأ
- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.333
  - GT:   الودود
  - Pred: الود
- `kuf14.jpg` — CER 0.706
  - GT:   الصبر مفتاح الفرج
  - Pred: الله جل جلاله

**3 worst (highest CER):**

- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 1.250
  - GT:   الله
  - Pred: الله أكبر
- `kuf19.jpg` — CER 1.100
  - GT:   نحلم ونحقق
  - Pred: يا حي يا قيوم
- `2_153.jpg` — CER 1.000
  - GT:   وذو حس فكاهي
  - Pred: فكلماتي ليغرس

##### Muhaqaq

**3 best (lowest CER):**

- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.000
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قل الله ثم ذرهم في خوضهم يلعبون
- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم

**3 worst (highest CER):**

- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.359
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين اهدنا الصراط
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.040
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العقل ومن شر حاسد اذا حسد
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.029
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمدلله رب العالمين الرحمن الرحيم

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.000
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفلق من شر ما خلق ومن شر
- `442 -1.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العالمين الرحمن الرحيم
- `1001-1.jpg` — CER 0.022
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: ويغفوا عن كثير قد جاءكم من الله نور وكتاب مبين

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.268
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: اعلنت لهم و اشرفت لهم أشرا رافقك استغفروا في
- `890-1.jpg` — CER 0.148
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: ليمتقن عندهم جنات النعيم افجعا المسلمين كالجرمين
- `778-3.jpg` — CER 0.111
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فسبعين شرهم إليه جميعا فأما الذين أمنوا وعملوا الصالحات

##### Nasta'liq

**3 best (lowest CER):**

- `nas47.jpg` — CER 0.298
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: لن تبلغ المجد حتى تلقي لصا كبيرا إلا تحسب المجدم أن تتشاركه
- `nas32.jpg` — CER 0.444
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: و على الذين شروا كل المتشوهون
- `nas49.jpg` — CER 0.630
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: اعْتِكافُ إِذَا عَلِمَ مَكَرَ صَدِقَ النَّبِيِّ

**3 worst (highest CER):**

- `nas51.jpg` — CER 2.636
  - GT:   نور علي نور
  - Pred: إِنَّا أَعْزُلْنَا عَمَّا سَأَرَى فَهِيَ مَا لَدَى رَبِّكُمْ
- `6_376.jpg` — CER 2.136
  - GT:   فباي الاء ربكما تكذبان
  - Pred: فَمَايَكُنْ مِّنْ آتٍِ ٱللَّهُ إِلَّا يُعْرِضُهُۥ لِأَحَدٍ فَإِنَّى أُنتَ عَزِيزٌ حَسْبِيَ ۚ وَشَانِئٌ كَثِيرٌ
- `nas35.jpg` — CER 1.158
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: المرء مع من أحب و يفرغ من نفسه

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.000
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عليه توكلت وإليه أنيب
- `ruq49.jpg` — CER 0.125
  - GT:   انت عمري
  - Pred: انت عمي
- `ruq26.jpg` — CER 0.278
  - GT:   يهلك المرء بأفكاره
  - Pred: يمهلك المرء أفقه

**3 worst (highest CER):**

- `ruq4.jpg` — CER 0.611
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط هندسة روحانية ظهرت بآلة جسمانية
- `ruq56.jpg` — CER 0.467
  - GT:   يارب أمور ميسرة
  - Pred: يا رب امني لسعت
- `ruq16.jpg` — CER 0.386
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: فكر كثيرا وتكلم قليلا افعل كثيرا واتكلم قليلا

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سُبحان الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.000
  - GT:   لكم دينكم ولي دين
  - Pred: لكم دينكم ولي دين
- `2_108.jpg` — CER 0.556
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله سُبحان الله وبحمده

**3 worst (highest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 2.250
  - GT:   الله
  - Pred: الله جل جلاله
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 0.852
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: و كان في قصصهم عبرة لأولى الألباب
- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 0.843
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: ومن كان يؤمن بالله واليوم الآخر فلا يؤذ جاره ومن كان يؤمن بالله واليوم الآخر فليكرم ضيفه

## ain-7b-routed

_Source: `calligraphy_ocr/results/zero_shot/ain-7b-routed_predictions.csv` (49 rows)_

- mean latency: **0.49s**, p95: **0.90s** per image
- peak VRAM observed: **16592 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.421 | 0.702 | 20.17 |
| Kufic | 0.770 | 1.000 | 21.43 |
| Muhaqaq | 0.028 | 0.071 | 85.82 |
| Naskh | 0.061 | 0.260 | 59.56 |
| Nasta'liq | 0.546 | 0.857 | 11.05 |
| Ruq'ah | 0.309 | 0.721 | 40.40 |
| Thuluth | 0.340 | 0.607 | 49.28 |
| **overall** | 0.353 | 0.603 | 41.10 |


#### Qualitative — ain-7b-routed


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.152
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: ويتولى الله بينما تظنّ أنك مفرورٌ
- `1643-Diwani1_segment_no_0.jpg` — CER 0.171
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئلون أهل الذكر إن كنتم تعلمون
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.200
  - GT:   ولذكر الله أكبر
  - Pred: ونذر الله البر

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.875
  - GT:   له الملك
  - Pred: الرحمن
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.805
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: والسلام من ربه توفيقي اني لا أزيد الا بذكر الله
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.468
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: ويستهنو الرسول والفضل وكان عليما حكما

##### Kufic

**3 best (lowest CER):**

- `kuf66.jpg` — CER 0.000
  - GT:   اقرأ
  - Pred: إقرأ
- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.333
  - GT:   الودود
  - Pred: الود
- `kuf14.jpg` — CER 0.706
  - GT:   الصبر مفتاح الفرج
  - Pred: الله جل جلاله

**3 worst (highest CER):**

- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 1.250
  - GT:   الله
  - Pred: الله أكبر
- `kuf19.jpg` — CER 1.100
  - GT:   نحلم ونحقق
  - Pred: يا حي يا قيوم
- `2_153.jpg` — CER 1.000
  - GT:   وذو حس فكاهي
  - Pred: فكلماتي ليغرس

##### Muhaqaq

**3 best (lowest CER):**

- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.000
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قل الله ثم ذرهم في خوضهم يلعبون
- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم

**3 worst (highest CER):**

- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.154
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين أهدينا الصراط المستقيم
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.040
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العقل ومن شر حاسد اذا حسد
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العالمين الرحمن الرحيم

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.000
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفلق من شر ما خلق ومن شر
- `442 -1.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العالمين الرحمن الرحيم
- `778-3.jpg` — CER 0.037
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فسيحشرهم إليه جميعا فأما الذين أمنوا وعملوا الصالحات

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.146
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أعلنت لهم وأشرت لهم أشرا إن فقلك استغفروا
- `1001-1.jpg` — CER 0.130
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: ويغفو اعز كثير قل جآءكم من الله نور و كتاب مبين
- `890-1.jpg` — CER 0.074
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: للمتقين عند ربهم جنات النعيم افتحعل المسلم كالمجرمين

##### Nasta'liq

**3 best (lowest CER):**

- `nas19.jpg` — CER 0.192
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: البَرِيْنِ وَجُدْطُلٌّ وكَلام لَيِّنٍ
- `nas49.jpg` — CER 0.259
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذا علم مبكرًا صدق الله أمانه
- `nas32.jpg` — CER 0.333
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وعلى الديب شوكل المثوكّون

**3 worst (highest CER):**

- `nas51.jpg` — CER 0.818
  - GT:   نور علي نور
  - Pred: تَوْابِعُ
- `nas47.jpg` — CER 0.754
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: لا تحسب المجدَمْ أنتَ اكْلُهُ
لن تبلغِ المجَد حَتَّى لعَقَصَ سيرًا
١٤ مشقه محبوبه
- `nas35.jpg` — CER 0.737
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: القمر يهمس في نسمة

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.000
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عليه توكلت وإليه أنيب
- `ruq49.jpg` — CER 0.125
  - GT:   انت عمري
  - Pred: انت عمي
- `ruq26.jpg` — CER 0.278
  - GT:   يهلك المرء بأفكاره
  - Pred: يمهلك المرء أفقه

**3 worst (highest CER):**

- `ruq4.jpg` — CER 0.611
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط هندسة روحانية ظهرت بآلة جسمانية
- `ruq56.jpg` — CER 0.467
  - GT:   يارب أمور ميسرة
  - Pred: يا رب امني لسعت
- `ruq16.jpg` — CER 0.386
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: فكر كثيرا وتكلم قليلا افعل كثيرا واتكلم قليلا

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سُبحان الله
- `Allah-1-Black_segment_no_0.jpg` — CER 0.000
  - GT:   الله
  - Pred: الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.059
  - GT:   لكم دينكم ولي دين
  - Pred: لكم دينكم ولي ديني

**3 worst (highest CER):**

- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 1.037
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: وكان في قلوبهم آناء من ينطق بالحق وهو موقوت
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 0.850
  - GT:   كل شيء هالك الا وجهه
  - Pred: كل شيء له علامة إلا أن تمحى عليه
- `2_108.jpg` — CER 0.278
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله سُبحان الله وتعالى

## ain-7b

_Source: `calligraphy_ocr/results/zero_shot/ain-7b_predictions.csv` (49 rows)_

- mean latency: **0.56s**, p95: **1.13s** per image
- peak VRAM observed: **16118 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.486 | 0.791 | 9.88 |
| Kufic | 0.839 | 1.048 | 12.10 |
| Muhaqaq | 0.135 | 0.324 | 46.82 |
| Naskh | 0.178 | 0.534 | 31.65 |
| Nasta'liq | 0.546 | 0.857 | 11.05 |
| Ruq'ah | 0.585 | 1.004 | 18.68 |
| Thuluth | 0.676 | 1.010 | 28.32 |
| **overall** | 0.492 | 0.795 | 22.64 |


#### Qualitative — ain-7b


##### Diwani

**3 best (lowest CER):**

- `1643-Diwani1_segment_no_0.jpg` — CER 0.143
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئلوا أهل الذكر إِن كنتم تعلمون
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.221
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: فاطرَ المحمّولات والأرض أنفس، وليّ في الدنيا وللآخرة توفّني مسماً وطغتي بالصّاحبين
- `dwn71.jpg` — CER 0.242
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: ويتولى الله بيئما نظن أنك مغرور

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 1.125
  - GT:   له الملك
  - Pred: الرحمن الرحيم
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.574
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: لهذا وانه خنوو والرموط والففريخ وكان عديم حكيمماً
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.561
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: وَالسِنْدُ وَإِلٰىهَا مِنْ أَصْلِ الْأَبْيَضِ

##### Kufic

**3 best (lowest CER):**

- `2_153.jpg` — CER 0.167
  - GT:   وذو حس فكاهي
  - Pred: وذو كلس فكاهي
- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.333
  - GT:   الودود
  - Pred: الود
- `kuf66.jpg` — CER 0.500
  - GT:   اقرأ
  - Pred: بِأَقْرَباً

**3 worst (highest CER):**

- `kuf19.jpg` — CER 1.800
  - GT:   نحلم ونحقق
  - Pred: بسم الله الرحمن الرحيم
- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 1.250
  - GT:   الله
  - Pred: الله أكبر
- `kuf84.jpg` — CER 1.000
  - GT:   لا تحزن
  - Pred: نعمنا

##### Muhaqaq

**3 best (lowest CER):**

- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.057
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الْحَمُدِ لَلّٰهِ رَبِّ الْعِلْمِينِ الرَّحمٰنِ الرَّحيمن

**3 worst (highest CER):**

- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.436
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وَإِلّا نَسْتَعِينُ أَهْدَنا الصِرَطَ
- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.226
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قُلِّ النَّبِيُّ ثُرَّرْهُم فِي خَوْضِهِم بَلٰعِبون
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.160
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: الْعُقَلِ وَمِن شَرِّ حاسِلاً ذَا حَسَدٌ

##### Naskh

**3 best (lowest CER):**

- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 0.037
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: سبْحَان رَبّكِ رَبِّ العِزَّةِ عَمّا يُصِفُونَ 0
- `442 -1.jpg` — CER 0.086
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الْحَمِدُ لِلّٰهِ رَبِّ الْعالَمينَ . أَلْرَحْمنَا الرَّحيْبهَ
- `Hasan-112-Small_segment_no_0.jpg` — CER 0.108
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ ﻫ  مِن شَيّراً خَلَقَه وَمِن شَرِّ

**3 worst (highest CER):**

- `778-3.jpg` — CER 0.296
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فَسَبْعَةُ وَهُوَ أَلِيه جَمِيعًا فَا مَا الّذِي اٮنوا وَعمَلاً والصَا ٮحان
- `890-1.jpg` — CER 0.278
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: لِأَمْتِيّز عند زُهُوم جَنَّان النَعيم افتخاعاً المُسلمين كاطرٮين
- `687-3.jpg` — CER 0.244
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أَعْلَنْتُ إِلهُم وَاشترىتُ لهُم أَسْرًا فَقُلْ اشتغفْهُما

##### Nasta'liq

**3 best (lowest CER):**

- `nas19.jpg` — CER 0.192
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: البَرِيْنِ وَجُدْطُلٌّ وكَلام لَيِّنٍ
- `nas49.jpg` — CER 0.259
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذا علم مبكرًا صدق الله أمانه
- `nas32.jpg` — CER 0.333
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وعلى الديب شوكل المثوكّون

**3 worst (highest CER):**

- `nas51.jpg` — CER 0.818
  - GT:   نور علي نور
  - Pred: تَوْابِعُ
- `nas47.jpg` — CER 0.754
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: لا تحسب المجدَمْ أنتَ اكْلُهُ
لن تبلغِ المجَد حَتَّى لعَقَصَ سيرًا
١٤ مشقه محبوبه
- `nas35.jpg` — CER 0.737
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: القمر يهمس في نسمة

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.048
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عَلَيْهِ نُوَكَّلْتُ وَإِلَيهِ أُنِيبُ
- `ruq26.jpg` — CER 0.278
  - GT:   يهلك المرء بأفكاره
  - Pred: بِهَلُك العَرْء بِأَفْطاط
- `ruq56.jpg` — CER 0.333
  - GT:   يارب أمور ميسرة
  - Pred: يا رب أَموميّة

**3 worst (highest CER):**

- `ruq53.jpg` — CER 1.185
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: بسم الله الرحمن الرحيم فلست بتغير العالم بحزنك
- `ruq49.jpg` — CER 1.000
  - GT:   انت عمري
  - Pred: ان شاء الله
- `ruq4.jpg` — CER 0.639
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط الهندسة روحانية ظهرت بألة جسمانية

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سبحان الله
- `2_108.jpg` — CER 0.222
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله هو سبحانه وتعالى
- `109-kaferon-white_segment_no_0.jpg` — CER 0.294
  - GT:   لكم دينكم ولي دين
  - Pred: لَكُمْ دِينكم، وَ لَهُ كُنٍّ

**3 worst (highest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 2.000
  - GT:   الله
  - Pred: الله هو الله
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 0.950
  - GT:   كل شيء هالك الا وجهه
  - Pred: كان بين يديه من الأدب ما لم تجده
- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 0.824
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: فليتقبل الأُولى بصمتٍ وَمن كان يؤمن بآبِدِيّهِ والْجُوُم الْأَخِرِ، ثم

## baseer-nakba

_Source: `calligraphy_ocr/results/zero_shot/baseer-nakba_predictions.csv` (49 rows)_

- mean latency: **0.23s**, p95: **0.41s** per image
- peak VRAM observed: **7427 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.548 | 0.839 | 10.24 |
| Kufic | 0.901 | 1.000 | 0.00 |
| Muhaqaq | 0.353 | 0.593 | 26.11 |
| Naskh | 0.310 | 0.695 | 17.79 |
| Nasta'liq | 0.660 | 0.964 | 4.23 |
| Ruq'ah | 0.594 | 0.879 | 12.70 |
| Thuluth | 0.546 | 0.671 | 30.85 |
| **overall** | 0.559 | 0.806 | 14.56 |


#### Qualitative — baseer-nakba


##### Diwani

**3 best (lowest CER):**

- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.286
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: فأطر الشعومان والأرض لأنهم. ولي في الدنيا والآخرة فوّني مسلماً وطمني أهلاً
- `dwn71.jpg` — CER 0.333
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: و ينتوتك الله بينما تظن انك
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.467
  - GT:   ولذكر الله أكبر
  - Pred: وذكر ابن كثير

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.875
  - GT:   له الملك
  - Pred: الثلاثاء
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.638
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: ونحنو وارسولوه للحضر وكان حكمت
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.634
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: و الله إن اول مرة أراني هذا الاسماعيليون

##### Kufic

**3 best (lowest CER):**

- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.667
  - GT:   الودود
  - Pred: الجمعة
- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 0.750
  - GT:   الله
  - Pred: السبت
- `kuf14.jpg` — CER 0.824
  - GT:   الصبر مفتاح الفرج
  - Pred: الحرية

**3 worst (highest CER):**

- `kuf66.jpg` — CER 1.250
  - GT:   اقرأ
  - Pred: الجمعة
- `kuf84.jpg` — CER 1.000
  - GT:   لا تحزن
  - Pred: الثلاثاء
- `2_153.jpg` — CER 0.917
  - GT:   وذو حس فكاهي
  - Pred: السبت

##### Muhaqaq

**3 best (lowest CER):**

- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Fatiha-Gold-Muhaqaq_segment_no_1.jpg` — CER 0.172
  - GT:   غير المغضوب عليهم ولا الضالين
  - Pred: غير المغصور عليه همه ولا الضالين
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.257
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله العلي العزيز الرحمن الرحيم

**3 worst (highest CER):**

- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.818
  - GT:   بسم الله الرحمن الرحيم
  - Pred: الثلاثاء
- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.538
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين عليك الصبر
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.360
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العقل ومن شرحك يا سيد

##### Naskh

**3 best (lowest CER):**

- `442 -1.jpg` — CER 0.143
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: اسمعه الله رب العالمين الرحمن الرحيم
- `Hasan-112-Small_segment_no_0.jpg` — CER 0.162
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفلك من شمسه ما خلق ومن نشرة
- `1001-1.jpg` — CER 0.283
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: و يضعفوا عز كثير قلب شجاعه الله نور و كتاب مبين

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.488
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أعلنت همة وانشراك ثم اشعرت فكفك ان شعفرها
- `778-3.jpg` — CER 0.463
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: وسيحشدهم كل الذين منسوا وعملوا إلى هنا.
- `890-1.jpg` — CER 0.333
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: للمتقبلة عنك تهز جنات النعيم فتجعل المشاهير كما مجيئ

##### Nasta'liq

**3 best (lowest CER):**

- `nas32.jpg` — CER 0.333
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: و على الله ينشدون كل المتسوكون
- `nas49.jpg` — CER 0.481
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذا لم مسنا رصد النسمة
- `nas47.jpg` — CER 0.684
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: اتمسيت بشع المجد حتى تلقعت الصبر

**3 worst (highest CER):**

- `nas51.jpg` — CER 0.909
  - GT:   نور علي نور
  - Pred: الجمعة
- `nas35.jpg` — CER 0.789
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: الثلاثاء
- `6_376.jpg` — CER 0.727
  - GT:   فباي الاء ربكما تكذبان
  - Pred: الاربعاء

##### Ruq'ah

**3 best (lowest CER):**

- `ruq53.jpg` — CER 0.259
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: فلن يتغير العالم بحزنك
- `ruq1.jpg` — CER 0.476
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عنكسة وَحدت وإِلَّا يُثيب
- `ruq26.jpg` — CER 0.611
  - GT:   يهلك المرء بأفكاره
  - Pred: جمالك الردفان

**3 worst (highest CER):**

- `ruq49.jpg` — CER 0.750
  - GT:   انت عمري
  - Pred: اعطا
- `ruq56.jpg` — CER 0.733
  - GT:   يارب أمور ميسرة
  - Pred: باريسو ٥
- `ruq4.jpg` — CER 0.667
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط هندسية روحانية ظرفت بان جسمانية

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سبحان الله
- `Allah-1-Black_segment_no_0.jpg` — CER 0.000
  - GT:   الله
  - Pred: الله
- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 0.392
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: ومنها يتوافر بالله واليوم الآخر فايق الامطار

**3 worst (highest CER):**

- `109-kaferon-white_segment_no_0.jpg` — CER 0.941
  - GT:   لكم دينكم ولي دين
  - Pred: الجمعة
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 0.852
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: الثلاثاء
- `2_108.jpg` — CER 0.833
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله

## ketaba-lora

_Source: `calligraphy_ocr/results/zero_shot/ketaba-lora_predictions.csv` (49 rows)_

- mean latency: **4.93s**, p95: **7.78s** per image
- peak VRAM observed: **8198 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.523 | 0.892 | 4.73 |
| Kufic | 0.819 | 1.000 | 0.00 |
| Muhaqaq | 0.291 | 0.630 | 28.59 |
| Naskh | 0.300 | 0.732 | 19.99 |
| Nasta'liq | 0.619 | 0.917 | 5.70 |
| Ruq'ah | 0.600 | 1.105 | 13.05 |
| Thuluth | 0.714 | 0.933 | 15.50 |
| **overall** | 0.552 | 0.887 | 12.51 |


#### Qualitative — ketaba-lora


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.303
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: وينودك الله بينما نظر أنك يغمرونك  God bless you while you think you are alone
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.325
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: فطر الأحمودان والارق أنسر ولي في أثرنا ولا فخرة نوفي مسمماً وأغمي بآصالتين
- `1643-Diwani1_segment_no_0.jpg` — CER 0.486
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسئوا أهل الأزرارباه ثم ادعموا

**3 worst (highest CER):**

- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.750
  - GT:   له الملك
  - Pred: الاحد
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.660
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: و سرعنون أرمو و لعذر فد كان عديم
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.600
  - GT:   ولذكر الله أكبر
  - Pred: وزراء اسبر الكير

##### Kufic

**3 best (lowest CER):**

- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.500
  - GT:   الودود
  - Pred: الاحد
- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 0.750
  - GT:   الله
  - Pred: الاحد
- `kuf19.jpg` — CER 0.800
  - GT:   نحلم ونحقق
  - Pred: الأولاء

**3 worst (highest CER):**

- `2_153.jpg` — CER 1.000
  - GT:   وذو حس فكاهي
  - Pred: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
- `kuf66.jpg` — CER 1.000
  - GT:   اقرأ
  - Pred: الاحد
- `kuf84.jpg` — CER 0.857
  - GT:   لا تحزن
  - Pred: الجمعة

##### Muhaqaq

**3 best (lowest CER):**

- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.200
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحماء لله رب العاملين الرمزا الرحيم
- `Fatiha-Gold-Muhaqaq_segment_no_1.jpg` — CER 0.276
  - GT:   غير المغضوب عليهم ولا الضالين
  - Pred: غير المغضوب عليهم هم و كل الضالين ٧

**3 worst (highest CER):**

- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.500
  - GT:   بسم الله الرحمن الرحيم
  - Pred: يشمس الله الحمراء
- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.385
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإياك نستعين أهدانا الصرط
- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.355
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قال الله نمذده في خوضهم يعبون ٧، ٣

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.027
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قل أعوذ برب الفرق من شر ما خلق ومن شر
- `442 -1.jpg` — CER 0.114
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الجمعة لله رب العالمين                الرحمان الرحيم
- `778-3.jpg` — CER 0.315
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فسيتحشروا إليه جميعاً فأما لذترا منوأ وعملاوا ألوانا الحان

**3 worst (highest CER):**

- `687-3.jpg` — CER 0.439
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: اعلنئ لمن واسرار امم اشترا فقائد استعمراري
- `890-1.jpg` — CER 0.407
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: للمستقر عند زهر جنائز الدعير افتحوا المسامين كما لم يجزر
- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 0.407
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: سبحان رب برك الله العزة عما يصفونه ٥

##### Nasta'liq

**3 best (lowest CER):**

- `nas49.jpg` — CER 0.296
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذا لم يكن قصدق النية  
أعاسارك
- `nas19.jpg` — CER 0.308
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: البرءين و به طددو وكلام ليرن
- `nas32.jpg` — CER 0.407
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وعلى الدفيء شول الممثوكون

**3 worst (highest CER):**

- `nas51.jpg` — CER 0.909
  - GT:   نور علي نور
  - Pred: الجمعة
- `nas47.jpg` — CER 0.842
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: إتحسب المجد ثم أنت اكؤ لن سبلغ المجد حتى تلقى لصبرا  ٤٤ مشق و بطولات    ١٤
- `nas35.jpg` — CER 0.842
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: الاحد

##### Ruq'ah

**3 best (lowest CER):**

- `ruq56.jpg` — CER 0.200
  - GT:   يارب أمور ميسرة
  - Pred: بارب امور مبشرة
- `ruq1.jpg` — CER 0.476
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عليه نوئ و إياه أذن
- `ruq4.jpg` — CER 0.611
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط هندسة روحانية ظهرت بأنه جسمانية

**3 worst (highest CER):**

- `ruq49.jpg` — CER 0.875
  - GT:   انت عمري
  - Pred: انشمر     بيروت
- `ruq53.jpg` — CER 0.815
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: ١٠ ر اس غرن فلس يغبر العالم بحزنك
- `ruq16.jpg` — CER 0.614
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: فكر كثيرون كلم قليلاً                 نكمل قليلا وفعل كثيراً

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سبحان الله
- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 0.314
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: ومن كان يوزن بالبر والبوم إلا آخر فليقعد الأولياء
- `2_108.jpg` — CER 0.667
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الللى                 أسود و يعاليه

**3 worst (highest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 1.250
  - GT:   الله
  - Pred: الثلاثاء
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 1.037
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: وَيْمَا رَأَى الْحُرُّ ١٧ الإِبْرَاهِيمِ وَلَهُمْ أَبْقَاءٌ
- `109-kaferon-white_segment_no_0.jpg` — CER 0.882
  - GT:   لكم دينكم ولي دين
  - Pred: الجمعة ٧         السبت                 الاثنين

## qari-v0.2.2.1

_Source: `calligraphy_ocr/results/zero_shot/qari-v0.2.2.1_predictions.csv` (49 rows)_

- mean latency: **0.69s**, p95: **1.41s** per image
- peak VRAM observed: **4688 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.922 | 1.333 | 0.94 |
| Kufic | 1.020 | 1.238 | 7.86 |
| Muhaqaq | 0.399 | 0.888 | 6.89 |
| Naskh | 1.438 | 1.917 | 5.17 |
| Nasta'liq | 0.578 | 0.952 | 2.77 |
| Ruq'ah | 0.588 | 1.029 | 3.35 |
| Thuluth | 0.531 | 0.786 | 21.43 |
| **overall** | 0.782 | 1.163 | 6.92 |


#### Qualitative — qari-v0.2.2.1


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.455
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: وَمِنْهُ لِلّ اللهِ بِبِنَا نَظِنُّ أَلنَّفأ محفروكِ
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.494
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: فَثرَ المُحمّد وَمِلْفَرْن فَلنَسٍ .وَيَبْتِي الارْشَا مِعَهُ عَمرَة نُوفَى مِساَحًا مُهْقَنًى بِاصْطَبِينِ
- `1643-Diwani1_segment_no_0.jpg` — CER 0.600
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فسيَّدُ زهْلِ الرَّتْرِ بِإِيمَامٍ نَعَامُورا 6

**3 worst (highest CER):**

- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 2.333
  - GT:   ولذكر الله أكبر
  - Pred: وَلِمَّا تُسْرِهِ بِك فِي أَبْدَارٍ مِّنْ كَفِّرِهِ وَلِأَبْدَارٍ مِّنْ كَفِّرِهِ
- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 1.250
  - GT:   له الملك
  - Pred: االسلا واسلام
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.683
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: وَاسْدِلْ مُنْتَهَى وَاشْمَرْ أَبْدَارٍ آسْفَانٌ

##### Kufic

**3 best (lowest CER):**

- `2_153.jpg` — CER 0.167
  - GT:   وذو حس فكاهي
  - Pred: وَذو كِس فكاهَاي
- `kuf66.jpg` — CER 0.500
  - GT:   اقرأ
  - Pred: لأقرنا
- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.500
  - GT:   الودود
  - Pred: الو 39 ٢

**3 worst (highest CER):**

- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 3.250
  - GT:   الله
  - Pred: اِلله اِللها الکاله
- `kuf84.jpg` — CER 1.000
  - GT:   لا تحزن
  - Pred: يُنِّيقِ
- `kuf19.jpg` — CER 0.900
  - GT:   نحلم ونحقق
  - Pred: يا لواءي

##### Muhaqaq

**3 best (lowest CER):**

- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.240
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: العَقْدُ وَمِنْسَرٍ حَاسِلًا أَحْسَد
- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.258
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قَالَ اللهُ مِنْهُم فَحُوضهم يلعبون
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.314
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله العليم الحمزالرحمن

**3 worst (highest CER):**

- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.591
  - GT:   بسم الله الرحمن الرحيم
  - Pred: للله الحرام
- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.513
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وَإِلّا نُسْتَحِين أهذَا الصرط
- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.500
  - GT:   بسم الله الرحمن الرحيم
  - Pred: مَالِ الله الحَمل بْهُم مَسٍّ

##### Naskh

**3 best (lowest CER):**

- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 0.222
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: سَجْنِ رُبَّكِ ربَ العَرَثة عَمَا نصعفون ٍ
- `Hasan-112-Small_segment_no_0.jpg` — CER 0.432
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قلَّ أُوعْدٍ بِرَبِّ الْفَلَقِيِّ وَمَن سَرِّهَاخَلَقْ
- `687-3.jpg` — CER 0.634
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أَعْلَنْتُ لِهَمُوّا شرَزٍ تَفَقُّدٌ؟ اسْتَغْبَضُوا كِبْدَكِ يَا أَيْمَانُ

**3 worst (highest CER):**

- `1001-1.jpg` — CER 6.500
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: رَمْسٍ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُورٍ رَمْسٌ مِّنْ سُبُور�
- `778-3.jpg` — CER 0.833
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: مَنْوَا وَعَالُوا لِلّا مَا الّذِي تَرَى جَمِيعًا فَأَمَّا هُورَةٌ بِهِ أَجْمَاعٍ يَتَّبَحُونَ
- `442 -1.jpg` — CER 0.743
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الرحْمَاتُ الحسَّةِ

##### Nasta'liq

**3 best (lowest CER):**

- `nas19.jpg` — CER 0.385
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: الحسن وجلاتون وكلام لين
- `nas32.jpg` — CER 0.407
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وَلِي الدُّفْتٍ مُوكَال الممْتَكِنون
- `nas49.jpg` — CER 0.407
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذاعلهم سبأ صدقة الله أعتاب

**3 worst (highest CER):**

- `nas51.jpg` — CER 0.818
  - GT:   نور علي نور
  - Pred: لادن
- `nas35.jpg` — CER 0.737
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: نَفْت نَفْت نَفْت
- `nas47.jpg` — CER 0.702
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: ارْتَحِسُ المِجرَّمَا أَنْكَاكِرَ مِثْلَهُ وَيَفْعَلُ الْجَدَّى لِقَبْضِ آخِرِهِ

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.238
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عَنِّيهٍ نُوَكْتُ وَإِيَّهُ أَنبُ
- `ruq49.jpg` — CER 0.500
  - GT:   انت عمري
  - Pred: اِنْتَهَى
- `ruq26.jpg` — CER 0.556
  - GT:   يهلك المرء بأفكاره
  - Pred: هَلكِ مُرْفَا فَطَا

**3 worst (highest CER):**

- `ruq16.jpg` — CER 0.750
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: فَكِرْ كُسْبٍ وَنَظَامٌ فَسِيدٌ رَطَّاحمٌ قَدِيرٌ وَافْعَل كُسْبٌ
- `ruq53.jpg` — CER 0.741
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: ش م فاسم بتفترا العالم كزنك
- `ruq56.jpg` — CER 0.667
  - GT:   يارب أمور ميسرة
  - Pred: بادآر سَنعة

##### Thuluth

**3 best (lowest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 0.000
  - GT:   الله
  - Pred: الله
- `6_357.jpg` — CER 0.100
  - GT:   سبحان الله
  - Pred: سبحانه الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.588
  - GT:   لكم دينكم ولي دين
  - Pred: لك مركوك وليمت ب

**3 worst (highest CER):**

- `2_108.jpg` — CER 0.833
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 0.778
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: لأناضِرُهُمْ مَعَكُمْ
- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 0.765
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: فَلِيقُ الْوَلاصُمِ وَكَايَ نُوبَانٍسُ البهَالخِ

## qari-v0.3

_Source: `calligraphy_ocr/results/zero_shot/qari-v0.3_predictions.csv` (49 rows)_

- mean latency: **1.29s**, p95: **4.05s** per image
- peak VRAM observed: **4477 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 3.253 | 4.238 | 2.52 |
| Kufic | 1.264 | 2.452 | 0.00 |
| Muhaqaq | 5.106 | 6.566 | 5.02 |
| Naskh | 1.842 | 2.934 | 7.05 |
| Nasta'liq | 1.339 | 1.738 | 2.64 |
| Ruq'ah | 0.732 | 1.306 | 6.81 |
| Thuluth | 1.344 | 1.910 | 15.61 |
| **overall** | 2.126 | 3.021 | 5.66 |


#### Qualitative — qari-v0.3


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.333
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: وينودك <i>الله</i> بينما نظن أنك مفروك <b>.مفروك</b> <u>.</u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.561
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: وَاسْتَأْكِمُ <b>وَاللَا</b> أَسْقَنَ اللَّهُ <i>بِهِ</i> <u></u>
- `10-At-Taghabun-64-1-Diwani_segment_no_0.jpg` — CER 0.625
  - GT:   له الملك
  - Pred: الله <i>الهاد</i>

**3 worst (highest CER):**

- `1643-Diwani1_segment_no_0.jpg` — CER 13.686
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: ا <b>اك</b> لاهد الله كله نعجهو طا فشكَهلابجكتم لا تسموه <i>نا</i> <u>م</u> <i>لا</i> <i>نهايته</i> <i>الله</i> لك <i>أنت</i> <i>ما</i> <i>كن</i> <i>لك</i> <i>نه</i> <i>بكم</i> <b>وك</b> <i>لنهايه</i> <i>للله</i> لك <i>أن</i> <i>تستمع</i> <i>نا</i> <i>من</i> <i>ك</i> لك <i>ان</i> <i>نعم</i> <i>لله</i> لك <i>ولهم</i> <i>لاتسموه</i> <b>نا</b> م <i>لا</i> هايته لله لك <i>نك</i> لنهايه لله لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i> لك نعمك <i>لاقتلاه</i
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 5.260
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: كما <b>وما</b> في المشابهلاً غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> المتشابهلاً غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <b>في</b> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِصَتْ لِبِينٍ فَمَا <i>الحمد</i> ك <i>وكي</i> <u>في</u> <i>المتشابهلاً</i> غمرة توفي مساحاً وَحُنثي بِص
- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 1.267
  - GT:   ولذكر الله أكبر
  - Pred: وَلِمْ أُسْتَحْرِب <b>وا</b> مَا كَانَهُ <i>أَكَامٌ</i> <u>.ا</u> <i>.</i>

##### Kufic

**3 best (lowest CER):**

- `2_153.jpg` — CER 0.583
  - GT:   وذو حس فكاهي
  - Pred: وَذِه كس فكاه <i>ما</i> ي؟
- `kuf14.jpg` — CER 0.706
  - GT:   الصبر مفتاح الفرج
  - Pred: الله <b>أعلم</b>
- `kuf66.jpg` — CER 0.750
  - GT:   اقرأ
  - Pred: ا <i>قرا</i> س

**3 worst (highest CER):**

- `kuf84.jpg` — CER 2.429
  - GT:   لا تحزن
  - Pred: يَا <b>مَن</b> يُعْتَلِّقُ بِكَأَنَّهُ مَرَّاً!
- `kuf19.jpg` — CER 1.800
  - GT:   نحلم ونحقق
  - Pred: لا <i>و</i> يا <b>لوق</b> <u>.ا</u> لامكوه <b>.</b>
- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 1.750
  - GT:   الله
  - Pred: ا <i>م</i> اك سه

##### Muhaqaq

**3 best (lowest CER):**

- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.258
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: “قَالَ <b>اللَّهُ</b> نَمْ ذَكَرَهُ مِنْ خَوْضِهِمْ لِيعبُونٍ”
- `Fatiha-Gold-Muhaqaq_segment_no_1.jpg` — CER 0.345
  - GT:   غير المغضوب عليهم ولا الضالين
  - Pred: "غَيْرُ <i>المخْضورِ</i> عَلِيهِ مَا لَأَكَالَّا يَنْتِهِ"
- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.436
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وإيَالك نستحِينُ <i>اهدنا</i> الصرْطٍ

**3 worst (highest CER):**

- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 18.636
  - GT:   بسم الله الرحمن الرحيم
  - Pred: مَا <i>لَهُ</i> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <b>الرحمنِ</b> الحِينَ مَا <i>لَهُ</i> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <b>الرحمنِ</b> الحِينَ مَا <i>لَهُ</i> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <b>الرحمنِ</b> الحِينَ مَا <b>لَهُ</b> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <b>الرحمنِ</b> الحِينَ مَا <b>لَهُ</b> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <b>الرحمنِ</b> الحِينَ مَا <u>لَهُ</u> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <u>الرحمنِ</u> الحِينَ مَا <u>لَهُ</u> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <u>الرحمنِ</u> الحِينَ مَا <u>لَهُ</u> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <u>الرحمنِ</u> الحِينَ مَا <u>لَهُ</u> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <u>الرحمنِ</u> الحِينَ مَا <u>لَهُ</u> الرِّحْمَنُ حِينَ لِسْ بِكِمَا اللهُ <u>الرحمنِ</u> الحِينَ مَا
- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 13.909
  - GT:   بسم الله الرحمن الرحيم
  - Pred: لِبَشْ <i>اللهُ</i> الرَّحِمَتُ مَا <b>يَكُنْ</b> لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُنْ لِهَا <u>الرَّحِمَتُ</u> مَا يَكُ
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 1.240
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: الْعُقَدِ وَمِنْشَرِّحٍ سَلِكٌ ذَا حَسَادٍ <b>ص</b> <i>ا</i> لقلِدِ <u>وَمِنْشَرِّحٍ</u> سَلِكٌ اذَا حَسَادٍ <i>)</i> <u></u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u> <u>)</u

##### Naskh

**3 best (lowest CER):**

- `Hasan-112-Small_segment_no_0.jpg` — CER 0.297
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: فُلْ أَعُوذٌ بِرَبِّيَا لَفَلَقِ <u>م</u> من شَرِكَاخَلَقَ <b>ة</b> وَمَن سَشِرِ <i></i>
- `442 -1.jpg` — CER 0.486
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: آمْحَدُ <i>لِهِ</i> رَبِّ الْعَالَمِينَ أَرْجَدٌ لِلهِ <b>رَبِّ</b> <i>الْعَالمِينَ</i> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u
- `687-3.jpg` — CER 0.512
  - GT:   أعلنت لهم وأسررت لهم إسرارا فقلت استغفروا
  - Pred: أَعْلَنْتُ لِهِمْ وَاشْرَدْ <i>ك</i> له م أَسْتَزِئْ ك</p><br><h3></h3>

**3 worst (highest CER):**

- `1001-1.jpg` — CER 6.783
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: وَمَا <i>يُسْتِدِّر</i> مَعَهُ أَغْزَةٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ <b>انْجَزَ</b> بَشَيرٌ <i>قَدَلَاجاً</i> كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْجَزَ بَشَيرٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْجَزَ بَشَيرٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْجَزَ بَشَيرٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْجَزَ بَشَيرٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْجَزَ بَشَيرٌ قَدَلَاجاً كَمْ مَازَالَهُ نَوْرًا وَكِبَابُ مُبَيِّضٍ لِأَنْفَهُ انْ
- `890-1.jpg` — CER 2.833
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: أَمْدَا، رِيدٍ <b>وَهُنَا</b> أَفْتَحَلُّ الْمَسْاجِرَكَالْجَيْشَ لِأَمْثِقَةٍ عِندَ تَبْهِمِهِمْ <i>ا</i> <u>.ان</u> <i>.</i> <b>وا</b> <i></i> <b>ع</b> . ا و</a> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b> <b>ا</b
- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 1.222
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: مَهْبُنٍ <i>دِبِّ</i> الْعِزَةِ عَمَا بِصِفُونَ 0 سَحْكَ رَتِلِكِ دِبِّ الْعِزَةِ <b>رَسَلِكِ</b> مَا يَصِفُونَ 0 <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u></u> <u

##### Nasta'liq

**3 best (lowest CER):**

- `nas32.jpg` — CER 0.333
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: و على الده فسيت موكك <i>المستوطكون</i>
- `nas49.jpg` — CER 0.481
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إذَا <b>الهَمِ</b> مُكَدٍّ صَدْوَ النِّيَةِ أَعَلَّاكَ، آيئاً
- `nas19.jpg` — CER 0.577
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: <p>الْبُسِّدَ، هُمُ <i>الثَّيِّنَ</i> ، وَجُهُطٍتَقَوَّ نَكَلامٌ لَيِينَ</p>

**3 worst (highest CER):**

- `nas51.jpg` — CER 5.727
  - GT:   نور علي نور
  - Pred: ءا <i>دم</i> . <b>م</b> نكينان كناله، تجاهنا <u>و</u> <i>أنت</i> ما <i>يقولون</i> <i>ما</i> <i>يقولون</i> <i>ما</i> <i>وقوا</i> <i>بكم</i> <i>.د</i> 1247 <i>-</i> <i>908</i> م <i>.</i>
- `nas47.jpg` — CER 0.842
  - GT:   لن تبْلغَ المجدَ حتّى تلعقَ الصَّبِرَا لا تحسبِ المجدَ تمرًا أنتَ آكلُهُ
  - Pred: الاتّسَبِ <i>المجْدُمَرَا</i> أشَكَاكُل ن بـنلعَ الهِدَحَتَي للعَقَصَة يبرا ٢٣ مشتقه <b>للجلخ</b> ١٤
- `6_376.jpg` — CER 0.727
  - GT:   فباي الاء ربكما تكذبان
  - Pred: كَمْكِنُ فَايٍّ <i>الآء</i> سُكِبَتْ SO، (O MANKIND AND JINN.) WHICH OF THE BOUNTIES OF YOUR LORD WILL YOU DENY? [QURAN 55:13]

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.190
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: عَلَيْهِ ثُوَكَّتُ وَاَنَبِيهِ أَنِيبٌ
- `ruq56.jpg` — CER 0.600
  - GT:   يارب أمور ميسرة
  - Pred: بادامودية
- `ruq26.jpg` — CER 0.667
  - GT:   يهلك المرء بأفكاره
  - Pred: هملك <i>المد</i> أ فطاع . / <u>شاكا</u> <b>.</b>

**3 worst (highest CER):**

- `ruq49.jpg` — CER 1.250
  - GT:   انت عمري
  - Pred: و انت كي <i>بأيا</i> د
- `ruq16.jpg` — CER 0.955
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: وَكَمْسِيَا وَنَظَاهُمْ قَسِيدًا <b>وَكَامَقَ</b> سَيهذًوا فَصِل كَشِيَا <i></i> <u>وَكَامَقَ</u> سَيهذًوا فَصِل كَشِيَا <b>)</b>
- `ruq53.jpg` — CER 0.741
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: م <i>اش</i> م فاسد يتفهّرَا لعالم كنزنك ا

##### Thuluth

**3 best (lowest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 0.000
  - GT:   الله
  - Pred: الله
- `2_108.jpg` — CER 0.611
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: الله <i>ه</i> <b>م</b> <u>ا</u> <i>وَتْحَايَاهُ</i>
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 1.100
  - GT:   كل شيء هالك الا وجهه
  - Pred: كَمِهَا <b>وَالله</b> إيمَدْ كَمِهَا وَالله <i>إيمَدْ</i>

**3 worst (highest CER):**

- `40Hadith-006-Thuluth_segment_no_1.jpg` — CER 3.098
  - GT:   ومن كان يؤمن بالله واليوم الآخر فليقل خيرا او ليصمت
  - Pred: الله <i>وله</i> <b>لأصمه</b> فليقك <u>ومن</u> <i>بنا</i> <u>سنه</u> والجهلاخا <u>ومنها</u> <i>يَمْتِدُ</i> <u>فَلنَا</u> <u>شَاءَ</u> <u>والجاه</u> لا يَحْذَرُ <i>في</i> <b>أن</b> نَعَمْ <b>لَاتَّضِيعُ</b> <b>بالله</b> <b>وكابئ</b> <b>وبني</b> <b>سبه</b> وثلما <b>ستكون</b> <b>كلمة</b> <b>غير</b> مألوفة في <b>لغة</b> العرب، فإنها <b>سيأتي</b> <i>به</i> <i>مع</i> <i>المعلومات</i> التي تريدها.
- `6_357.jpg` — CER 1.800
  - GT:   سبحان الله
  - Pred: سجَدْ <i>لِه</i> الله سورة <b>النور</b> <i>(Nur)</i> <u>.و</u> <u>.</u>
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 1.444
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: وَكَالْهِجَا <b>لِسَمُّ</b> هَا كَانَتْهَا مَا لَاتُهَا <i>أَنَصَرَ</i> أَيْدٌ <u>ا</u> نضَاءً <b>للشَّامِقِ</b> ا <u>.></u>

## sherif-handwritten-v3

_Source: `calligraphy_ocr/results/zero_shot/sherif-handwritten-v3_predictions.csv` (49 rows)_

- mean latency: **0.36s**, p95: **0.85s** per image
- peak VRAM observed: **7427 MB**


### Per-style metrics (lower CER/WER, higher BLEU is better)

| style | cer | wer | bleu |
|---|---|---|---|
| Diwani | 0.542 | 0.905 | 10.81 |
| Kufic | 1.501 | 1.310 | 3.93 |
| Muhaqaq | 0.326 | 0.681 | 24.10 |
| Naskh | 0.216 | 0.562 | 31.92 |
| Nasta'liq | 0.684 | 0.976 | 8.28 |
| Ruq'ah | 0.767 | 1.224 | 16.26 |
| Thuluth | 0.697 | 1.019 | 15.81 |
| **overall** | 0.676 | 0.954 | 15.87 |


#### Qualitative — sherif-handwritten-v3


##### Diwani

**3 best (lowest CER):**

- `dwn71.jpg` — CER 0.212
  - GT:   ويتولاك الله بينما تظن أنك بمفردك
  - Pred: وينزل الله بينما 
تظن انك 
يغردك
God bless you while you think
you are alone
- `Yusuf-12-101-Diwani_segment_no_0.jpg` — CER 0.403
  - GT:   فاطر السماوات والأرض أنت وليي في الدنيا والآخرة توفني مسلما وألحقني بالصالحين
  - Pred: قطرل المعمّدون والامر من الله . ولي قي البرنادى الافرة توفى مسلمًا فطمني بالصاتحين
- `1643-Diwani1_segment_no_0.jpg` — CER 0.571
  - GT:   فاسألوا أهل الذكر إن كنتم لا تعلمون
  - Pred: فَشْلَوَهُمْ ذِرَابٌ لَّن مِعَلْمُوا

**3 worst (highest CER):**

- `003-Al-Ankabut-29-45_segment_no_0.jpg` — CER 0.667
  - GT:   ولذكر الله أكبر
  - Pred: ولنزر النبي الله بهر
- `03-Al-Fath-48-1to4-Diwani_segment_no_5.jpg` — CER 0.660
  - GT:   ولله جنود السماوات والأرض وكان الله عزيزا حكيما
  - Pred: وينعمتو والارحومه ولله درقها كان عديمي
- `40Hadith-005-DiwaniJelli-229x300_segment_no_0.jpg` — CER 0.659
  - GT:   والله لا يؤمن والله لا يؤمن والله لا يؤمن
  - Pred: وَلَا تَرْكُوهُمْ لِلنَّاسِ الْخَيْلِينَ

##### Kufic

**3 best (lowest CER):**

- `2_153.jpg` — CER 0.250
  - GT:   وذو حس فكاهي
  - Pred: وذا لب فكاهي
- `Al-Wadud-Kufic-Black-300x300_segment_no_0.jpg` — CER 0.667
  - GT:   الودود
  - Pred: الفرنك
- `kuf19.jpg` — CER 0.800
  - GT:   نحلم ونحقق
  - Pred: نائم

**3 worst (highest CER):**

- `Basmah-Allah-Kufipsd__segment_no_0.jpg` — CER 5.250
  - GT:   الله
  - Pred: المملكة الحجازية اليمانية
- `kuf84.jpg` — CER 1.714
  - GT:   لا تحزن
  - Pred: عضو نعمه فرحات
- `kuf66.jpg` — CER 1.000
  - GT:   اقرأ
  - Pred: kusara

##### Muhaqaq

**3 best (lowest CER):**

- `Falaq-Muhaqaq_segment_no_3.jpg` — CER 0.000
  - GT:   بسم الله الرحمن الرحيم
  - Pred: بسم الله الرحمن الرحيم
- `Fatiha-Gold-Muhaqaq_segment_no_4.jpg` — CER 0.200
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمد لله رب العملين ان حمنا السجيم
- `Falaq-Muhaqaq_segment_no_0.jpg` — CER 0.200
  - GT:   العقد ومن شر حاسد إذا حسد
  - Pred: الْعُقَلِ وَمِنْ شَرٍّ حَايِئٍ أَذَا جَسَدك

**3 worst (highest CER):**

- `Basmallah-6-White-300x119_segment_no_0.jpg` — CER 0.636
  - GT:   بسم الله الرحمن الرحيم
  - Pred: ليش نعمل االحمل الحميري
- `Al-Anam-691-300x57_segment_no_0.jpg` — CER 0.581
  - GT:   قل الله ثم ذرهم في خوضهم يلعبون
  - Pred: قَالُوا لَهُمْ ۗ وَلَا تُدْرِي مَا خَصْضُوهُمْ بِعِبُولٍ
- `Fatiha-Gold-Muhaqaq_segment_no_3.jpg` — CER 0.462
  - GT:   وإياك نستعين اهدنا الصراط المستقيم صراط
  - Pred: وَإِيّا كنستعين هالنا الصُرْط

##### Naskh

**3 best (lowest CER):**

- `442 -1.jpg` — CER 0.000
  - GT:   الحمد لله رب العالمين الرحمن الرحيم
  - Pred: الحمدُ لله رب العالمين 	الرحمن الرحيم
- `Al-Safat-37-180-182-naskh2_segment_no_2.jpg` — CER 0.074
  - GT:   سبحن ربك رب العزة عما يصفون
  - Pred: سبحان ربك رب العزة عما يصفونك
- `Hasan-112-Small_segment_no_0.jpg` — CER 0.162
  - GT:   قل أعوذ برب الفلق من شر ما خلق  ومن شر
  - Pred: قُلْ أَعُوذُ بِرَبِّ الْفَلَקَ * رِيذ شَتّى مَا خلق وَمَن شَرٍّ

**3 worst (highest CER):**

- `778-3.jpg` — CER 0.407
  - GT:   فسيحشرهم إليه جميعا فأما الذين ءامنوا و عملوا الصالحات
  - Pred: فَسَيْتِجُزُّوهُمْ أَلَيهِ جَمِيعًا ذَا قَائِماً لِذِرَاءِ مَنٌو وَعَمَلاً ۖ إِنَّهَا الْحَانَ
- `890-1.jpg` — CER 0.370
  - GT:   للمتقين عند ربهم جنات النعيم أفنجعل المسلمين كالمجرمين
  - Pred: لله وَمِنْقِذٍ عَلَى ذُهُمْ جِنّاتِ النَّعيمِ أَفَجَعلَا لِلنَّاسِ مِيئًا كَامِجِرًتِين
- `1001-1.jpg` — CER 0.304
  - GT:   ويعفوا عن كثير قد جاءكم من الله نور وكتاب مبين
  - Pred: وَيُعْفِوهُ إِذَا غَرَّ كَثِيرٌ قَدْ جَاءَهُ مِنَ اللَّهِ نُورًا وَجَابًا ۖ وَمُبِينًأ

##### Nasta'liq

**3 best (lowest CER):**

- `nas32.jpg` — CER 0.370
  - GT:   وعلى الله فليتوكّل المتوكّلون
  - Pred: وعلى الدين يستولى الممتوكون
- `nas49.jpg` — CER 0.593
  - GT:   إذا علِمَ منكَ صِدْقَ النية أعَانك
  - Pred: إِذَٰلَكَ مِن نُرْصُدَ النَّيَةِ 
أعاسارىك آيق
- `nas19.jpg` — CER 0.654
  - GT:   البِرُّ هَيِّن وَجهُ طَلقُ وكَلامُ لَيِّنُ
  - Pred: البرّ، والحسَن ، 
وبحوثِهُ وكتابَمْ لٌبنٍّ

**3 worst (highest CER):**

- `nas35.jpg` — CER 0.842
  - GT:   اَلْقَنَاعَةُ كَنْزٌ لاَ يُفْنى
  - Pred: الخوري ىعوم درىا ١٦
- `nas51.jpg` — CER 0.818
  - GT:   نور علي نور
  - Pred: قيراط 
١٦
- `6_376.jpg` — CER 0.773
  - GT:   فباي الاء ربكما تكذبان
  - Pred: الاسَمى وكنزل 
قاي

##### Ruq'ah

**3 best (lowest CER):**

- `ruq1.jpg` — CER 0.333
  - GT:   عَلَيْهِ تَوَكَّلْتُ وَإِلَيْهِ أُنِيبُ
  - Pred: غَلبهُ ذوقتٌ و إلهه أنيب
- `ruq16.jpg` — CER 0.591
  - GT:   تكلم قليلاً وأفعل كثيراً فكر كثيراً وتكلم قليلاً
  - Pred: فكر كثيراً وتكلم قليلاً 
تكلم قليلاً وافعل كثيرًا
- `ruq4.jpg` — CER 0.667
  - GT:   ظهرت بآلة جسمانية الخط هندسة روحانية
  - Pred: الخط هندسة روحيانية
ظهرت بألا جسَمانيّة

**3 worst (highest CER):**

- `ruq49.jpg` — CER 1.500
  - GT:   انت عمري
  - Pred: اعري  انتحرر جبرايل
- `ruq53.jpg` — CER 0.889
  - GT:   فلن يتغيّ العالم بحزنك ابتسم
  - Pred: من انت غنيه 
فلن بقية العام . كذنك
- `ruq26.jpg` — CER 0.722
  - GT:   يهلك المرء بأفكاره
  - Pred: عهلك الرد بأفكاره 
٣١/١٢/٨٥٤

##### Thuluth

**3 best (lowest CER):**

- `6_357.jpg` — CER 0.000
  - GT:   سبحان الله
  - Pred: سبحان الله
- `109-kaferon-white_segment_no_0.jpg` — CER 0.471
  - GT:   لكم دينكم ولي دين
  - Pred: لي كم يتنكر ولي الحين
- `2_108.jpg` — CER 0.667
  - GT:   ٱللَّهُ سُبْحَانَهُ وَتَعَالَى
  - Pred: اللهُ سَمّيَتِ اللهُهَ وَبَعْدًا

**3 worst (highest CER):**

- `Allah-1-Black_segment_no_0.jpg` — CER 1.000
  - GT:   الله
  - Pred: 垄陇块块
- `al-rum-30-47-thuluth-02_segment_no_0.jpg` — CER 1.000
  - GT:   وكان حقا علينا نصر المؤمنين
  - Pred: وَهُمْ بِغَيْرِ مَا يَعْلَمُونَ * وَإِذَا قَبَّلَ ٧٢ رَسْلًا
- `3-Qasas-28-88-Thuluth-2_segment_no_1.jpg` — CER 1.000
  - GT:   كل شيء هالك الا وجهه
  - Pred: ابنة محمد بن ادريس الافحل