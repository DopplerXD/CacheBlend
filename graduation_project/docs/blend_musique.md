# blend_musique.py 说明

MusiQue 主实验脚本，在 Musique 数据集上比较 `full_prefill`、`full_reuse`、`query_aware` 三种处理方式，并把逐样本结果与整体汇总写入结构化日志。

## 数据格式

读取 json 数据，数据集路径 `inputs/musique_s.json`

- `question`：问题文本
- `answers`：标准答案列表
- `ctxs`：文档列表
  - `title`
  - `text`

```json
{
    "ctxs": [
        {
            "title": "",
            "text": "community's commitment to a solution in haiti and discussed the establishment of a new police force in haiti under a proposed united nations mission in haiti ( unmih ). saburro peak is a peak in antarctica to the south of doll peak in the south part of ravens mountains, britannia range. named after colonel richard m. saburro, commander, operation deep freeze who was assigned to this position from the 109 ag of the new york air national guard. saburro was the first us air force commander of operation deep freeze following 45 years of command by the us navy. the operation provides military logistics support to the national science foundation's us antarctic program. garden state life insurance company is a small direct life insurance company located in league city, texas. it is a wholly owned subsidiary of the galveston, texas based american national insurance company. selway national forest was established by the u. s. forest service in idaho on july 1, 1911 with from parts of clearwater national forest and nez perce national forest. on october 29, 1934 the entire forest was divided between bitterroot, clearwater, lolo and nez perce, and the name was discontinued. william barret \" buck \" travis ( august 1, 1809 \u2013 march 6, 1836 ) was a 19th - century american lawyer and soldier. at the age of 26, he was a lieutenant colonel in the texas army. he died at the battle of the alamo during the texas revolution. travis county and travis park were named after him for being the commander of the republic of texas at the battle of the alamo. patrick lavon mahomes ii ( born september 17, 1995 ) is an american football quarterback for the kansas city chiefs of the national football league ( nfl ). he played college football at texas tech, and was drafted by the chiefs with the tenth overall pick in the 2017 nfl draft. mahomes is the son of former mlb pitcher pat mahomes. the texas revolution ( october 2, 1835 - - april 21, 1836 ) was a rebellion of colonists from the united states and tejanos ( texas mexicans ) in putting up armed resistance to the centralist government of mexico. while the uprising was part of a larger one that included other provinces opposed to the regime of president antonio lopez de santa anna, the mexican government believed the united states had instigated the texas insurrection with the goal of annexation. the mexican congress passed the tornel decree, declaring that any foreigners fighting against mexican troops ` ` will be deemed pirates and"
        },
        {
            "title": "",
            "text": "software which had come bundled with a video capture card he'd purchased a few years earlier. facebook and myspace were used to gather actors to play the zombies. during 1889, thomas edison had business interests in many electricity - related companies : edison lamp company, a lamp manufacturer in east newark, new jersey ; edison machine works, a manufacturer of dynamos and large electric motors in schenectady, new york ; bergmann & company, a manufacturer of electric lighting fixtures, sockets, and other electric lighting devices ; and edison electric light company, the patent - holding company and the financial arm backed by j. p. morgan and the vanderbilt family for edison's lighting experiments. in 1889, drexel, morgan & co., a company founded by j. p. morgan and anthony j. drexel, financed edison's research and helped merge those companies under one corporation to form edison general electric company which was incorporated in new york on april 24, 1889. the new company also acquired sprague electric railway & motor company in the same year. beaver motorcoach corporation ( also known as beaver coach ) is a defunct american motor coach manufacturing company that was based in oregon. the company's manufacturing plant was initially located in bend and later moved to coburg. after its initial bankruptcy, the beaver coach brand name was purchased by a series of parent companies before it finally disappeared in 2009. fuji heavy industries started out as the aircraft research laboratory in 1915, headed by chikuhei nakajima. in 1932, the company was reorganized as nakajima aircraft company, ltd and soon became a major manufacturer of aircraft for japan during world war ii. at the end of the second world war nakajima aircraft was again reorganized, this time as fuji sangyo co, ltd. in 1946, the company created the fuji rabbit motor scooter with spare aircraft parts from the war. in 1950, fuji sangyo was divided into 12 smaller corporations according to the japanese government's 1950 corporate credit rearrangement act, anti - zaibatsu legislation. between 1953 and 1955, four of these corporations and a newly formed corporation decided to merge to form fuji heavy industries. these companies were : fuji kogyo, a scooter manufacturer ; coachbuilders fuji jidosha ; engine manufacturers omiya fuji kogyo ; chassis builders utsunomiya sharyo and the tokyo fuji dangyo trading company. a nigerian state is a federated political entity, which shares sovereignty with the federal government of nigeria, there"
        }
    ],
    "question": "In which county is Southern Maryland Electric Cooperative headquartered?",
    "answers": [
        "Charles County"
    ]
},
```

输入：
- 一条多文档问答样本

输出：
- 该样本在不同方法下的生成结果和统计指标，最后汇总统计结果均值。

```json
{
    "event": "run_start",
    "script": "example/blend_musique.py",
    "dataset": "inputs/musique_s.json",
    "started_at": "2026-04-15 21:21:26",
    "model": "/root/models/Yi-6B",
    "max_new_tokens": 32,
    "temperature": 0,
    "top_p": 1
}

{
    "event": "sample_result",
    "sample_idx": 14,
    "chunk_num": 10,
    "question": "The Unwinding author volunteered for which organisation?",
    "answers": [
        "Peace Corps"
    ],
    "stale_cache_prompt_tokens": 5884,
    "full_reuse": {
        "generated_text": "peace corps",
        "ttft_s": 1.0063338539998767,
        "total_s": 1.0988792240000294,
        "reused_prefix_tokens": 5877,
        "recompute_mode": "cache_prefix_reuse",
        "f1": 1
    },
    "query_aware": {
        "generated_text": "the Peace Corps",
        "ttft_s": 1.0131146999997327,
        "total_s": 1.134766987999683,
        "reused_prefix_tokens": 0,
        "recomputed_tokens": 4123,
        "recompute_mode": "query_aware_recompute",
        "true_recompute": true,
        "f1": 1
    },
    "full_prefill": {
        "generated_text": "peace corps",
        "ttft_s": 1.356506970999817,
        "total_s": 1.448365449999983,
        "f1": 1
    }
}

{
    "event": "run_summary",
    "sample_count": 30,
    "full_reuse_avg_ttft_s": 1.1018977398332785,
    "query_aware_avg_ttft_s": 1.0688906501333348,
    "full_prefill_avg_ttft_s": 1.4696762840999782,
    "full_reuse_avg_total_s": 1.2991096991333204,
    "query_aware_avg_total_s": 1.3038722372000089,
    "full_prefill_avg_total_s": 1.66684074399991,
    "full_reuse_avg_f1": 0.2672005772005772,
    "query_aware_avg_f1": 0.25338025808614045,
    "full_prefill_avg_f1": 0.2672005772005772,
    "qaw_true_recompute_count": 30,
    "ended_at": "2026-04-15 21:25:20"
}
```

## 运行流程

### 0 运行方式

```bash
python example/blend_musique.py
python example/blend_musique.py --count 50 --qaw-ratio 0.7
python example/blend_musique.py --output-dir outputs/useful
```

这个脚本真正暴露给命令行的参数只有三个：

- `--count`：最多评测多少条样本，默认 `30`
- `--qaw-ratio`：`query_aware` 的重算比例，默认 `0.7`
- `--output-dir`：日志输出目录，默认 `outputs`

### 1 运行环境初始化

创建 `RuntimeConfig`，将 `max_new_tokens` 固定为 `32`，然后基于配置初始化：

- `HFModelRunner`
- `KVCacheManager`
- `InferenceEngine`
- `ExperimentOutputWriter`

其中 `ExperimentOutputWriter` 默认把日志写到 `outputs/*_musique.output`。脚本启动后先写一条 `run_start`，记录模型名、温度、`count`、`qaw_ratio` 等基本配置信息。

### 2 数据处理

是逐样本顺序进行，每条样本先通过 `build_qa_prompt()` 把原始样本整理成：

- `doc_prompts`：文档块列表
- `q_prompt`：问题部分 prompt

再构造两个完整输入：

- `final_prompt`：真实问答使用的 prompt
- `stale_prompt`：旧缓存预热使用的占位 prompt

这里的 `stale_prompt` 和真实问题不同，目的是先制造一份“已有缓存但与当前问题不完全匹配”的旧缓存状态，从而让 `query_aware` 真正处于“旧缓存 + 新问题”的重算场景中。

### 3 顺序执行三种方法

首先执行 `full_prefill`。这一阶段关闭缓存复用，直接对 `final_prompt` 做完整 prefill，再继续生成答案。它对应的请求设置是：

- `use_cache=False`
- `recompute_strategy="none"`

这一结果作为最基础的 baseline，用来对比其他方法在时延和质量上的变化。

接着执行 `full_reuse`。脚本会先把同一个 `final_prompt` 预填充到一个模板 session 中，再把这个 session 的 KV 复制到运行 session，随后在 `use_cache=True`、`recompute_strategy="none"` 下继续生成。这个方法的重点是观察“完全命中前缀缓存”时的收益。

最后执行 `query_aware`。也是先从 `stale_prompt` 产生的旧缓存出发，但重算策略改为 `query_aware`。这里的关键设置是：

- `use_cache=True`
- `recompute_strategy="query_aware"`
- `recomp_ratio=args.qaw_ratio`
- `suffix_len=32`
- `query_text=query_text`

### 4 汇总输出结果

从每种结果中提取同一组核心指标：

- 生成文本 `generated_text`
- 首 token 时延 `ttft_s`
- 总时延 `total_s`
- 复用的前缀 token 数 `reused_prefix_tokens`（有的策略会记录）
- 重算的 token 数 `recomputed_tokens`（`query_aware` 会记录）
- 实际执行模式 `recompute_mode`
- `F1`
- 是否发生真实重算 `true_recompute`

 将单条样本的结果写成一条 `sample_result`：

- sample_idx
- chunk_num
- question
- answers
- stale_cache_prompt_tokens
- full_reuse
- query_aware
- full_prefill
    - generated_text
    - ttft_s
    - total_s
    - f1


所有样本处理完成后，汇总取平均，输出一条 `run_summary`：

- sample_count
- full_reuse_avg_ttft_s
- query_aware_avg_ttft_s
- full_prefill_avg_ttft_s
- full_reuse_avg_total_s
- query_aware_avg_total_s
- full_prefill_avg_total_s
- full_reuse_avg_f1
- query_aware_avg_f1
- full_prefill_avg_f1
- qaw_true_recompute_count
- ended_at
