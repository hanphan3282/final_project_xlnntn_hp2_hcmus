nextflow.enable.dsl = 2

params.from_stage = 1
params.to_stage = 8
params.reuse_existing = true
params.data_dir = "${projectDir}/data"
params.env_file = "${projectDir}/.env"
params.provider = 'deepseek'
params.limit = null

params.fetch_workers = 24
params.fetch_retries = 4
params.gemini_workers = 4
params.gemini_retries = 4
params.paddle_workers = 2
params.paddle_cpu_threads = 4
params.llm_workers = 4
params.llm_retries = 4
params.compare_threshold = 0.58
params.min_common_han = 4
params.min_confidence = 0.75

def enabled(int stage) {
    def first = params.from_stage as int
    def last = params.to_stage as int
    return stage >= first && stage <= last
}

def optionalLimit() {
    return params.limit == null ? '' : "--limit ${params.limit as int}"
}

process FETCH_IMAGES {
    label 'network_io'
    cache false

    input:
    val previous

    output:
    path 'stage_01.json', emit: done

    script:
    def limitArg = optionalLimit()
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 1 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --reuse-existing '${params.reuse_existing}' --marker stage_01.json -- \
      python '${projectDir}/scripts/1_fetch_images.py' \
        --input '${params.data_dir}/input/valid.jsonl' \
        --images-dir '${params.data_dir}/input/Images' \
        --error-log '${params.data_dir}/output/mrDuc_data_ocr/fetch_errors.jsonl' \
        --summary '${params.data_dir}/output/mrDuc_data_ocr/fetch_summary.json' \
        --workers ${params.fetch_workers} --retries ${params.fetch_retries} \
        --report-every 100 ${limitArg}
    """

    stub:
    "touch stage_01.json"
}

process GEMINI_OCR {
    label 'network_io'
    cache false

    input:
    val previous

    output:
    path 'stage_02.json', emit: done

    script:
    def limitArg = optionalLimit()
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 2 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --reuse-existing '${params.reuse_existing}' --marker stage_02.json -- \
      python '${projectDir}/scripts/2_ocr_gemini.py' \
        --input '${params.data_dir}/input/valid.jsonl' \
        --images-dir '${params.data_dir}/input/Images' \
        --output-jsonl '${params.data_dir}/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl' \
        --output-json '${params.data_dir}/output/mrDuc_data_ocr/facebook_posts_ocr.json' \
        --error-log '${params.data_dir}/output/mrDuc_data_ocr/ocr_errors.jsonl' \
        --summary '${params.data_dir}/output/mrDuc_data_ocr/ocr_summary.json' \
        --env-file '${params.env_file}' --workers ${params.gemini_workers} \
        --retries ${params.gemini_retries} --timeout 240 --report-every 50 ${limitArg}
    """

    stub:
    "touch stage_02.json"
}

process COMPARE_LABELS {
    label 'light'
    cache false

    input:
    val previous

    output:
    path 'stage_03.json', emit: done

    script:
    def limitArg = optionalLimit()
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 3 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --reuse-existing '${params.reuse_existing}' --marker stage_03.json -- \
      python '${projectDir}/scripts/3_compare_gemini_label.py' \
        --labels '${params.data_dir}/input/valid.jsonl' \
        --gemini '${params.data_dir}/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl' \
        --images-dir '${params.data_dir}/input/Images' \
        --same-dir '${params.data_dir}/output/Gemini_same_Label' \
        --diff-dir '${params.data_dir}/output/Gemini_diff_Label' \
        --threshold ${params.compare_threshold} --min-common-han ${params.min_common_han} ${limitArg}
    """

    stub:
    "touch stage_03.json"
}

process PADDLE_OCR {
    label 'cpu_ocr'
    cache false

    input:
    val previous

    output:
    path 'stage_04.json', emit: done

    script:
    def limitArg = optionalLimit()
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 4 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --reuse-existing '${params.reuse_existing}' --marker stage_04.json -- \
      python '${projectDir}/scripts/4_paddlev6_relabel_diff.py' \
        --input '${params.data_dir}/output/Gemini_diff_Label/records.jsonl' \
        --output-dir '${params.data_dir}/output/Gemini_diff_Label/paddle_v6' \
        --workers ${params.paddle_workers} --cpu-threads ${params.paddle_cpu_threads} \
        --score-threshold 0.30 --fallback-confidence 0.65 ${limitArg}
    """

    stub:
    "touch stage_04.json"
}

process LLM_ADJUDICATION {
    label 'network_io'
    cache false

    input:
    val previous

    output:
    path 'stage_05.json', emit: done

    script:
    def limitArg = optionalLimit()
    def outputName = params.provider == 'deepseek' ? 'DeepSeek_ground_truth' : 'Qwen38_ground_truth'
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 5 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --provider '${params.provider}' --reuse-existing '${params.reuse_existing}' \
      --marker stage_05.json -- \
      python '${projectDir}/scripts/5_llm_adjudicate_ground_truth.py' \
        --provider '${params.provider}' \
        --labels '${params.data_dir}/input/valid.jsonl' \
        --gemini '${params.data_dir}/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl' \
        --diff '${params.data_dir}/output/Gemini_diff_Label/records.jsonl' \
        --paddle '${params.data_dir}/output/Gemini_diff_Label/paddle_v6/new_labels.jsonl' \
        --images-dir '${params.data_dir}/input/Images' \
        --output-dir '${params.data_dir}/output/${outputName}' \
        --env-file '${params.env_file}' --workers ${params.llm_workers} \
        --retries ${params.llm_retries} --timeout 240 --max-tokens 8192 \
        --report-every 50 ${limitArg}
    """

    stub:
    "touch stage_05.json"
}

process SPLIT_ADJUDICATIONS {
    label 'light'
    cache false

    input:
    val previous

    output:
    path 'stage_06.json', emit: done

    script:
    def outputName = params.provider == 'deepseek' ? 'DeepSeek_ground_truth' : 'Qwen38_ground_truth'
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 6 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --provider '${params.provider}' --reuse-existing '${params.reuse_existing}' \
      --marker stage_06.json -- \
      python '${projectDir}/scripts/split_adjudications.py' \
        --input '${params.data_dir}/output/${outputName}/adjudications.jsonl' \
        --valid '${params.data_dir}/output/${outputName}/adjudications_valid.jsonl' \
        --invalid '${params.data_dir}/output/${outputName}/adjudications_invalid.jsonl' \
        --min-confidence ${params.min_confidence}
    """

    stub:
    "touch stage_06.json"
}

process BUILD_GROUND_TRUTH {
    label 'light'
    cache false

    input:
    val previous

    output:
    path 'stage_07.json', emit: done

    script:
    def outputName = params.provider == 'deepseek' ? 'DeepSeek_ground_truth' : 'Qwen38_ground_truth'
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 7 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --provider '${params.provider}' --reuse-existing '${params.reuse_existing}' \
      --marker stage_07.json -- \
      python '${projectDir}/scripts/6_ground_truth.py' \
        --input '${params.data_dir}/output/${outputName}/adjudications_valid.jsonl' \
        --output '${params.data_dir}/output/${outputName}/ground_truth.jsonl' \
        --provider '${params.provider}'
    """

    stub:
    "touch stage_07.json"
}

process COPY_GROUND_TRUTH_IMAGES {
    label 'light'
    cache false

    input:
    val previous

    output:
    path 'stage_08.json', emit: done

    script:
    def outputName = params.provider == 'deepseek' ? 'DeepSeek_ground_truth' : 'Qwen38_ground_truth'
    """
    python '${projectDir}/workflow/run_stage.py' \
      --stage 8 --project-dir '${projectDir}' --data-dir '${params.data_dir}' \
      --provider '${params.provider}' --reuse-existing '${params.reuse_existing}' \
      --marker stage_08.json -- \
      python '${projectDir}/scripts/copy_ground_truth_images.py' \
        --ground-truth '${params.data_dir}/output/${outputName}/ground_truth.jsonl' \
        --src '${params.data_dir}/input/Images' \
        --dst '${params.data_dir}/ground_truth_images'
    """

    stub:
    "touch stage_08.json"
}

workflow {
    if ((params.from_stage as int) < 1 || (params.to_stage as int) > 8 ||
        (params.from_stage as int) > (params.to_stage as int)) {
        error "Khoảng stage không hợp lệ: ${params.from_stage}..${params.to_stage}; cần 1 <= from_stage <= to_stage <= 8"
    }
    if (!(params.provider in ['deepseek', 'qwen'])) {
        error "provider phải là deepseek hoặc qwen"
    }

    def token = Channel.value('pipeline-start')
    if (enabled(1)) token = FETCH_IMAGES(token)
    if (enabled(2)) token = GEMINI_OCR(token)
    if (enabled(3)) token = COMPARE_LABELS(token)
    if (enabled(4)) token = PADDLE_OCR(token)
    if (enabled(5)) token = LLM_ADJUDICATION(token)
    if (enabled(6)) token = SPLIT_ADJUDICATIONS(token)
    if (enabled(7)) token = BUILD_GROUND_TRUTH(token)
    if (enabled(8)) token = COPY_GROUND_TRUTH_IMAGES(token)

    token.view { "Pipeline hoàn tất: ${it}" }
}
