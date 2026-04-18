# CMRC2018数据集

https://github.com/ymcui/cmrc2018

当前仓库中的 CMRC 实验输入不是直接把原始数据集喂给实验脚本，而是先从 `CMRC2018` 验证集抽样并转换成统一格式，再保存为 `inputs/cmrc_s.json`。

`CMRC2018` 是中文机器阅读理解数据集，任务形式是给定一段中文上下文和一个问题，要求模型从上下文中回答问题。和当前仓库里常用的 `musique`、`wikimqa` 相比，CMRC 的几个特点比较明显：

- 是中文问答任务
- 大多数样本只有一段主要上下文，而不是多文档组合
- 上下文整体显著更短
- 问题和答案也更短，形式更接近传统阅读理解

当前实验并没有直接使用完整原始验证集，而是从验证集里固定抽样 50 条，做成 `inputs/cmrc_s.json`。这样做的目的主要是和现有实验脚本保持一致，方便直接套用同一套日志与统计方式。

原始数据来源路径在预处理脚本中写死为：

```text
/Users/doppler/documents/github/cmrc2018/squad-style-data/cmrc2018_dev.json
```

对应的预处理脚本是：

```text
scripts/preprocess_cmrc2018.R
```

这个脚本的作用不是训练，也不是建模，而是把原始验证集整理成当前项目统一使用的输入格式。

它的处理流程比较直接：

1. 读取 `cmrc2018_dev.json`
2. 按原始层级展开：
   - `data`
   - `paragraphs`
   - `qas`
3. 对每个问答项抽取：
   - `context`
   - `question`
   - `answers`
4. 对答案做去重，并过滤空答案
5. 把每条样本改写成当前项目使用的统一结构
6. 用固定随机种子抽样 50 条
7. 输出到 `inputs/cmrc_s.json`

这里用到的固定随机种子是：

```text
2026
```

默认抽样数量是：

```text
50
```

默认输出路径是：

```text
inputs/cmrc_s.json
```

预处理后的 `cmrc_s.json` 与当前项目其他实验输入保持同样的字段组织方式。单条样本包含：

- `question`
- `answers`
- `ctxs`

其中 `ctxs` 是一个列表，每个元素包含：

- `title`
- `text`

对 CMRC 来说，`ctxs` 通常只有一项，也就是单文档形式。这样就能让 `blend_cmrc.py`、`blend_curve.py --dataset cmrc`、`blend_suffix_curve.py --dataset cmrc` 直接复用现有 prompt 构造和日志输出逻辑，而不用重新设计一套新的输入接口。

可以把 `cmrc_s.json` 理解成“把原始 CMRC2018 dev 中适合实验的样本抽出来，并适配成与 `musique_s.json`、`wikimqa_s.json` 同类的统一实验输入格式”。

常用预处理命令如下：

```bash
Rscript scripts/preprocess_cmrc2018.R
```

如果需要显式指定输入、输出、抽样数或随机种子，也可以这样运行：

```bash
Rscript scripts/preprocess_cmrc2018.R \
  --input-json /Users/doppler/documents/github/cmrc2018/squad-style-data/cmrc2018_dev.json \
  --output-json inputs/cmrc_s.json \
  --max-samples 50 \
  --seed 2026
```

从实验角度看，CMRC 与 `musique / wikimqa` 的最大差异不在日志格式，而在数据形态：

- `musique` 和 `wikimqa` 更偏多文档长上下文问答
- `cmrc` 更偏单文档短上下文中文阅读理解

因此，`cmrc_s.json` 的引入主要是为了在保持实验框架统一的前提下，增加一个中文、短上下文、单文档的对照数据集。
