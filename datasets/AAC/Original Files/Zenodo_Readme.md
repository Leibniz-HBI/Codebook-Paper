# Argument Aspect Corpus
The Argument Aspect Corpus (AAC) contains argumentative English-language sentences from four
different topics with aspect annotations on a token level. It was introduced in
this paper:

> Mattes Ruckdeschel and Gregor Wiedemann. 2022. Boundary Detection and
Categorization of Argument Aspects via Supervised Learning. In Proceedings of
the 9th Workshop on Argument Mining, pages 126–136, Online and in Gyeongju,
Republic of Korea. International Conference on Computational Linguistics.

The Corpus is based on the argumentative sentences in the UKP SAM[1] dataset for
four highly debated topics: nuclear energy, minimum wage, abortion, and marijuana
legalization. The corpus contains one conll-formatted file per topic, containing the gold
standard annotation. Further the coding guidelines used for annotation are uploaded.
of all sentences from that topic. For the reproduction of paper results, check out
the corresponding [GitHub repository][repo].
The gold standard annotation was obtained by *chunk-normalization* of a
token-level gold standard. Using the default chunker from flair[2], sentences
were split into chunks, and all tokens of a chunk were labeled with an aspect if
at least one token in the chunk was labeled. Any conflicts were resolved by an additional
coder.

Coding was done by two trained expert coders with a
background in Social science. Conflicts were resolved by a third trained coder
with a background in Computer Science.

## Topic information
The following tables shows statistics for the different topics. $\alpha_k$ gives
the intercoder-agreement as Krippendorff's alpha. *Arg Occurrences* gives the
number of arguments containig a specific aspect, while *Chunk Occurrences* gives
the number chunks that have been labeled with a specific aspect.

**General Statistics**
$N_{args}$ describes the number of arguments for a topic and $N_{single}$ the
amount of arguments with only one aspect.

|        **Topic**       | $N_{args}$ | $N_{single}$ |
|:--------------------------:|:----------:|:------------:|
|     Minimum Wage (MW)      |    1118    |      938     |
|   Nuclear Energy (NE)      |    1261    |      992     |
| Marijuana Legalization (MJ)|    1213    |     1006     |
|        Abortion   (AB)     |    1502    |     1305     |


**Minimum Wage**

|            **Aspect**            | $\alpha_k$ | Arg Occurrences | Chunk Occurrences |
|:--------------------------------:|:----------:|:--------------------:|:-----------------:|
|        Un/employment rate        |    0.80    |          259         |        287        |
|        Motivation/chances        |    0.67    |          86          |        107        |
|  Competition/business challenges |    0.58    |          104         |        129        |
|              Prices              |    0.88    |          93          |        104        |
|     Social justice/injustice     |    0.70    |          305         |        353        |
|              Welfare             |    0.76    |          49          |         57        |
|          Economic impact         |    0.80    |          81          |         99        |
|             Turnover             |    0.96    |          22          |         32        |
|         Capital vs labour        |    0.51    |          25          |         32        |
|            Government            |    0.65    |          38          |         71        |
|            Low-skilled           |    0.69    |          85          |        100        |
| Youth and secondary wage earners |    0.58    |          24          |         37        |
|               Other              |    0.56    |          160         |        160        |
|            all topics            |    0.65    |         1331         |        1568       |

**Nuclear Energy**

|        **Aspect**        | $\alpha_k$ | Arg Occurrences | Chunk Occurrences |
|:------------------------:|:----------:|:--------------------:|:-----------------:|
|           Waste          |    0.80    |          121         |        152        |
|      Health effects      |    0.67    |          100         |        128        |
|   Environmental impact   |    0.58    |          236         |        313        |
|           Costs          |    0.88    |          131         |        170        |
|          Weapons         |    0.70    |          60          |         66        |
|        Reliability       |    0.76    |          106         |        134        |
| Technological innovation |    0.80    |          59          |         79        |
|       Energy policy      |    0.96    |          99          |        135        |
|        Renewables        |    0.51    |          121         |        143        |
|       Fossil fuels       |    0.65    |          99          |        120        |
|    Accidents/security    |    0.69    |          270         |        365        |
|       Public debate      |    0.58    |          47          |         75        |
|           Other          |    0.56    |          139         |        139        |
|        all topics        |    0.65    |         1585         |        2017       |


**Marijuana Legalization**

|          **Aspect**          | $\alpha_k$ | Arg Occurrences | Chunk Occurrences |
|:----------------------------:|:----------:|:--------------------:|:-----------------:|
|         Illegal trade        |    0.87    |          100         |        130        |
|     Child and teen safety    |    0.89    |          124         |        149        |
|  Community/Societal effects  |    0.54    |          153         |        196        |
| Health/Psychological effects |    0.78    |          188         |        302        |
|       Medical Marijuana      |    0.92    |          134         |        183        |
|          Drug abuse          |    0.78    |          66          |         78        |
|           Addiction          |    0.95    |          59          |         72        |
|       Personal freedom       |    0.79    |          41          |         54        |
|        National budget       |    0.77    |          114         |        154        |
|         Gateway drug         |    0.90    |          47          |         60        |
|          Legal drugs         |    0.91    |          108         |        130        |
|          Drug policy         |    0.50    |          104         |        137        |
|             Harm             |    0.53    |          77          |         94        |
|             Other            |    0.49    |          139         |        139        |
|          all topics          |    0.64    |         1454         |        1879       |


**Abortion**

|                **Aspect**               | $\alpha_k$ | Arg Occurrencens | Chunk Occurrences |
|:---------------------------------------:|:----------:|:----------------:|:-----------------:|
|      Bodily autonomy/Women's rights     |    0.57    |        267       |        385        |
|           Fetal/newborn rights          |    0.83    |        507       |        719        |
|                   Rape                  |    0.96    |        49        |         59        |
|            Abortion industry            |    0.84    |        15        |         18        |
|           Moral/ethical values          |    0.67    |        139       |        173        |
| Safety/health effects of legal abortion |    0.81    |        88        |        113        |
|    Psychological effects of abortion    |    0.84    |        60        |         78        |
|  Health effects of pregnancy/childbirth |    0.75    |        95        |        116        |
|            Illegal abortions            |    0.83    |        54        |         75        |
|              Responsibility             |    0.64    |        59        |         81        |
|                 Adoption                |    0.93    |        39        |         44        |
|        Consequences of childbirth       |    0.66    |        96        |        130        |
|        Fetal defects/disabilities       |    0.90    |        47        |         60        |
|             Parental consent            |    0.80    |        16        |         25        |
|           Funding of abortions          |    0.70    |        20        |         25        |
|                  Other                  |    0.48    |        172       |        172        |
|                all topics               |    0.66    |       1723       |        2273       |




[1] Stab, C., Miller, T., Schiller, B., Rai, P., & Gurevych, I. Cross-topic
Argument Mining from Heterogeneous Sources. In E. Riloff, D. Chiang, J.
Hockenmaier, & J. Tsujii (Eds.), Proceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing (pp. 3664–3674). Association for
Computational Linguistics. https://doi.org/10.18653/v1/D18-1402

[2] Alan Akbik, Tanja Bergmann, Duncan Blythe, Kashif Rasul, Stefan Schweter, and Roland Vollgraf. 2019. FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 54–59, Minneapolis, Minnesota. Association for Computational Linguistics.

[repo]:https://github.com/Leibniz-HBI/argument-aspect-corpus-v1

