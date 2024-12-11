import sys
import requests
from pathlib import Path
import pandas as pd
import json
import ast
import yaml
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from loguru import logger
import click
from tqdm import tqdm
import re
from transformers import AutoTokenizer


logger.remove()

logger.add(sys.stderr, level="DEBUG")

@click.group()
def cli():
    pass



@cli.command()
@click.argument('path', required=True)
@click.option('-c', '--create', is_flag=True, help='runs create before starting experiments')
def run(path, create):
    click.echo('running experiments')
    run_all_experiments(path)


def mean_df(list_of_dfs):
    result_df = pd.DataFrame(index=list_of_dfs[0].index, columns=list_of_dfs[0].columns)
    for col in result_df.columns:
        for idx in result_df.index:
            values = [df.loc[idx, col] for df in list_of_dfs]
            mean = np.mean(values)
            std = np.std(values)
            result_df.loc[idx, col] = f"{mean:.2f} ± {std:.2f}"
    return result_df


def generate(prompt, context, model, output_info=True):
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                      },
                      stream=True)
    r.raise_for_status()
    response = ''
    for line in r.iter_lines():
        body = json.loads(line)
        if not body.get('done'):
            response += body.get('response', '')
        else:
            if output_info:
                response_dict =  {'response':response}
                response_dict.update({k: v for k, v in body.items()  if k not in ['context','response','done']})
                return response_dict
            else:
                return response
        # the response streams one token at a time, print that as we receive it
        if 'error' in body:
            raise Exception(body['error'])
    logger.info('something wrong with this test case here')
    logger.info(r)
    if output_info:
        return {'response':'no info','info':'no info'}
    else:
        return 'no info'

def get_all_labels(column):
    all_labels = set()
    for label_list in column:
        for label in label_list:
            all_labels.add(label)
    return list(all_labels)



def count_labels(response, labels, expand_labels=True):
    extended_labels = {}
    mentioned_labels = []
    for label in labels:
        extended_labels[label] = [label]
        if expand_labels:
            if '_' in label:
                extended_labels[label].append(label.replace('_',' '))
            ### Hardcoded edge cases in minimum wage dataset
            if label == 'COMPETITION/BUSINESS_CHALLENGES':
                extended_labels[label].append('COMPETITION')
                extended_labels[label].append('BUSINESS_CHALLENGES')
                extended_labels[label].append('BUSINESS CHALLENGES')
            if label == 'UN/EMPLOYMENT_RATE':
                extended_labels[label].append('EMPLOYMENT_RATE')
                extended_labels[label].append('EMPLOYMENT RATE')
            if label == 'MOTIVATION/CHANCES':
                extended_labels[label].append('MOTIVATION')
                extended_labels[label].append('CHANCES')
            if label == 'SOCIAL_JUSTICE/INJUSTICE':
                extended_labels[label].append('SOCIAL_JUSTICE')
                extended_labels[label].append('SOCIAL JUSTICE')
                extended_labels[label].append('SOCIAL INJUSTICE')
                extended_labels[label].append('SOCIAL_INJUSTICE')
            if label == 'ENVIRONMENTAL_IMPACT':
                extended_labels[label].append('Environmental Impact')
            if label == 'SAFETY/HEALTH_EFFECTS_OF_LEGAL_ABORTION':
                extended_labels[label].append('HEALTH EFFECTS OF LEGAL ABORTION')
                extended_labels[label].append('HEALTH_EFFECTS_OF_LEGAL_ABORTION')
                extended_labels[label].append('SAFETY')
    for label, label_expressions in extended_labels.items():
        for expression in label_expressions:
            if expression in response:
                mentioned_labels.append(label)
                break
    return len(mentioned_labels), mentioned_labels


def generate_mlb_classification_report(pred, gold, labels_as_list):
    """
    Generates a classification report for multilabel classification tasks.

    Parameters:
    pred (list of lists): Predicted labels.
    gold (list of lists): Ground truth labels.

    Returns:
    pd.DataFrame: Classification report as a pandas DataFrame with string labels.
    """
    mlb = MultiLabelBinarizer()
    if not labels_as_list:
        if isinstance(gold[0], str): ## then that has to be cast to lists
            gold = pd.Series([[label] for label in gold])
        if isinstance(pred[0],str):
            pred = pd.Series([[label] for label in pred])
    mlb.fit(gold + pred)
    y_true_bin = mlb.transform(gold)
    y_pred_bin = mlb.transform(pred)
    report = classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True)
    report_df = pd.DataFrame(report).T
    return report_df



def test_from_csv(
        file, model, outpath,
        prompt_prefix='', batch_size=1,
        text_column_index=0, label_column=1,
        labels_as_list=True, num_of_icl_examples=0, icl_file_path=''):
    df = pd.read_csv(file)
    if labels_as_list:
        df[df.columns[label_column]] = df[df.columns[label_column]].apply(ast.literal_eval)
    if labels_as_list:
        all_labels = get_all_labels(df[df.columns[label_column]])
    else:
        all_labels = list(df[df.columns[label_column]].unique()) # casting to list just to be safe!
    replace_text = False
    if num_of_icl_examples > 0:
        logger.info(f'adding {num_of_icl_examples} to the prompt')
        sample_df = pd.read_csv(icl_file_path)
        logger.info(f'loaded samples from {icl_file_path}')
        if 'reasons' in sample_df.columns:
            ICL_template = ('Input:\n<SAMPLE>\nOutput:\n<REASON>','Input:\n<TEXT>\nOutput:\n')
        else:
            sample_df.labels = sample_df.labels.apply(ast.literal_eval)
            sample_df.labels = sample_df.labels.apply(lambda x: x[0])
            sample_df = sample_df[sample_df.labels != 'OTHER']
            ICL_template = ('Input:\n<SAMPLE>\nOutput:\nThe correct label is <LABEL>\n','Input:\n<TEXT>\nOutput:\n')
        template_text = prompt_prefix + '\n'
        for label in all_labels:
            logger.info(f'adding for {label}')
            if label != 'OTHER': #don't take any examples from OTHER
                for row in range(0, num_of_icl_examples):
                    sample_text = sample_df[sample_df.labels == label].iloc[row]['sentence']
                    if 'reasons' in sample_df.columns:
                        sample_reason = sample_df[sample_df.labels == label].iloc[row]['reasons'] #yeah it's plural...
                    to_add = ICL_template[0].replace('<SAMPLE>', sample_text)
                    if 'reasons' in sample_df.columns:
                        to_add = to_add.replace('<REASON>', sample_reason)
                    to_add = to_add.replace('<LABEL>', label)
                    template_text += to_add
        template_text += ICL_template[1]
        logger.debug(template_text)
        replace_text = True
    if replace_text:
        responses = [generate(template_text.replace('<TEXT>', sentence), [], model) for sentence in tqdm(df[df.columns[text_column_index]],desc='generating')]
    else:
        responses = [generate(prompt_prefix + sentence, [], model) for sentence in tqdm(df[df.columns[text_column_index]],desc='generating')]
    results_df = pd.DataFrame(responses)
    results_df['input_text'] = df[df.columns[text_column_index]]
    results_df['gold_label'] = df[df.columns[label_column]]
    results_df['label_count'], results_df['mentioned_labels'] = zip(*results_df['response'].apply(lambda x: count_labels(x, all_labels)))
    no_other = results_df[results_df.gold_label.apply(lambda x: 'OTHER' not in x)].copy()
    # sometimes there's OTHER mentioned by the LLM without knowledge of the label, leading to some problems
    no_other.mentioned_labels = no_other.mentioned_labels.apply(lambda x: [label for label in x if label != 'OTHER'])
    results_df.to_csv(f'{outpath}/results.csv', index=False)
    results_df['label_count'].value_counts().to_csv(f'{outpath}/num_label_counts.csv')
    no_other['label_count'].value_counts().to_csv(f'{outpath}/num_label_counts_no_other.csv')
    report_df = generate_mlb_classification_report(results_df.mentioned_labels, results_df.gold_label, labels_as_list)
    report_df.to_csv(f'{outpath}/classification_report.csv')
    report_df_otherless = generate_mlb_classification_report(no_other.mentioned_labels, no_other.gold_label, labels_as_list)
    report_df_otherless.to_csv(f'{outpath}/classification_report_no_other.csv')
    logger.info(f'saved results from {file} with {model} in {outpath}')


def read_config(path):
    with open(path / 'config.yml', 'r') as cfg:
        config = yaml.safe_load(cfg)
    logger.debug(f'loaded config from {path}')
    return config


def analyze_results_in_subfolders(base_path, dataset_name):
    """
    Iterate through all subfolders of the given base path, find 'results.csv' files,
    and call the ollama_experiments.analyze_results_file(file) function on each.

    Parameters:
    base_path (str or Path): The base directory path to start the search.
    """
    base_path = Path(base_path)  # Ensure base_path is a Path object
    test_file = pd.read_csv(f'datasets/AAC/{dataset_name}_original_test.csv')
    test_file.labels = test_file.labels.apply(ast.literal_eval)
    all_labels_ds = set()
    for label_list in test_file.labels:
        for label in label_list:
            all_labels_ds.add(label)
    label_list = list(all_labels_ds)
    for file_path in base_path.rglob('results.csv'):
        analyze_results_file(file_path, label_list ,dataset_name)
        print(f'Analyzed: {file_path}')


def analyze_results_file(file, label_list, dataset_name):
    if isinstance(file,str):
        file = Path(file)
    results_df = pd.read_csv(file)
    test_file = pd.read_csv(f'datasets/AAC/{dataset_name}_original_test.csv')
    test_file.labels = test_file.labels.apply(ast.literal_eval)
    if 'info' in results_df.columns:
        logger.info('fixing old file')
        results_df['info'] = results_df['info'].apply(ast.literal_eval)
        df_expanded = results_df['info'].apply(lambda x: {k: v for k, v in x.items() if k not in ['context','response','done']}).apply(pd.Series)
        results_df = pd.concat([results_df.drop(columns=['info']), df_expanded], axis=1)
        results_df['label_count'], results_df['mentioned_labels'] = zip(*results_df['response'].apply(lambda x: count_labels(x, label_list)))
        results_df['gold_label'] = test_file['labels']
        results_df['input_text'] = test_file['sentence']
        logger.info('saving updated_file')
        results_df.to_csv(file.parent / 'modified_results_file.csv')
    else:
        results_df['mentioned_labels'] = results_df['mentioned_labels'].apply(ast.literal_eval)
        results_df['gold_label'] = results_df['gold_label'].apply(ast.literal_eval)
    no_other = results_df[results_df.gold_label.apply(lambda x: 'OTHER' not in x)]
    results_df.to_csv(f'{file.parent}/results.csv', index=False)
    results_df['label_count'].value_counts().to_csv(f'{file.parent}/num_label_counts.csv')
    no_other['label_count'].value_counts().to_csv(f'{file.parent}/num_label_counts_no_other.csv')
    report_df = generate_mlb_classification_report(results_df.mentioned_labels, results_df.gold_label)
    report_df.to_csv(f'{file.parent}/classification_report.csv')
    report_df_otherless = generate_mlb_classification_report(no_other.mentioned_labels, no_other.gold_label)
    report_df_otherless.to_csv(f'{file.parent}/classification_report_no_other.csv')


def additional_stats_from_results(file, outfile=True, no_no_theme=False, no_other=False):
    if isinstance(file,str):
        file = Path(file)
    results_df = pd.read_csv(file)
    results_df.mentioned_labels = results_df.mentioned_labels.apply(ast.literal_eval)
    results_df.gold_label = results_df.gold_label.apply(ast.literal_eval)
    if no_other:
        results_df = results_df[results_df.gold_label.apply(lambda x: 'OTHER' not in x)].copy()
        results_df.mentioned_labels = results_df.mentioned_labels.apply(lambda x: [label for label in x if label != 'OTHER'])
        results_df.label_count = results_df.mentioned_labels.apply(len)
    if no_no_theme:
        results_df = results_df[results_df.gold_label.apply(lambda x: 'NO_THEME' not in x)].copy()
        results_df.mentioned_labels = results_df.mentioned_labels.apply(lambda x: [label for label in x if label != 'NO_THEME'])
        results_df.label_count = results_df.mentioned_labels.apply(len)
        if not (file.parent / 'no_no_classification_report.csv').exists():
            nono_report = generate_mlb_classification_report(gold=results_df.gold_label, pred=results_df.mentioned_labels, labels_as_list=True)
            nono_report.to_csv(file.parent / 'no_no_classification_report.csv')
    stats = []
    hit_percentage = len(results_df[results_df.apply(lambda x: all(elem in x['mentioned_labels'] for elem in x['gold_label']), axis=1)]) / len(results_df)
    single_hit = len(results_df[results_df.apply(lambda x: any(elem in x['mentioned_labels'] for elem in x['gold_label']), axis=1)]) / len(results_df)
    stats.append({'stat': 'full_hit_percentage', 'value': hit_percentage})
    stats.append({'stat': 'single_hit_percentage', 'value': single_hit})
    stats.append({'stat': 'mean_labels_assigned', 'value': results_df.label_count.mean()})
    stats.append({'stat': 'std_labels_assigned', 'value':results_df.label_count.std()})
    stats.append({'stat': 'no_labels_assigned', 'value': len(results_df[results_df.label_count == 0])})
    num_labels = len(get_all_labels(results_df.gold_label))
    stats.append({'stat': 'all_labels_assigned', 'value': len(results_df[results_df.label_count == num_labels])})
    stats.append({'stat': 'mean_total_duration', 'value': results_df.total_duration.mean() / 1_000_000_000}) # total duration is in ns
    stats.append({'stat': 'mean_response_len', 'value': results_df.response.str.len().mean()})
    stats.append({'stat': 'std_response_len', 'value': results_df.response.str.len().std()})
    stat_df = pd.DataFrame(stats)
    if outfile:
        stat_df.to_csv(file.parent / 'additional_stats.csv')
    return stat_df








def process_reports(base_path):
    if isinstance(base_path,str):
        base_path = Path(base_path)

    # Iterate through each dataset/model directory
    for model_dir in base_path.glob('*/*/'):
        logger.debug(f'processing {model_dir}')
        classification_reports = []
        classification_reports_no_other = []
        classification_reports_no_no = []
        stats = []

        # Iterate through the numbered subdirectories (1, 2, 3, 4, 5)
        for numbered_dir in model_dir.glob('[1-5]'):
            logger.debug(f'processing {numbered_dir}')
            # Read classification_report.csv
            report_path = numbered_dir / 'classification_report.csv'
            if report_path.exists():
                classification_reports.append(pd.read_csv(report_path, index_col=0))

            # Read classification_report_no_other.csv
            report_no_other_path = numbered_dir / 'classification_report_no_other.csv'
            if report_no_other_path.exists():
                classification_reports_no_other.append(pd.read_csv(report_no_other_path, index_col=0))

            report_no_no_path = numbered_dir / 'no_no_classification_report.csv'
            if report_no_no_path.exists():
                classification_reports_no_no.append(pd.read_csv(report_no_no_path, index_col=0))

            additional_stats_path = numbered_dir / 'additional_stats.csv'
            if additional_stats_path.exists():
                stats.append(pd.read_csv(additional_stats_path, index_col=0, usecols=[1,2]))


        # Calculate mean_df for both reports if we have data
        if classification_reports:
            logger.debug('creating classification_report')
            mean_report = mean_df(classification_reports)
            mean_report.to_csv(model_dir / 'report_summary.csv')

        if classification_reports_no_other:
            logger.debug('creating classification_report without OTHERS')
            mean_report_no_other = mean_df(classification_reports_no_other)
            mean_report_no_other.to_csv(model_dir / 'report_summary_no_other.csv')


        if classification_reports_no_no:
            logger.debug('creating classification_report without NOs')
            mean_report_no_no = mean_df(classification_reports_no_no)
            mean_report_no_no.to_csv(model_dir / 'report_summary_no_no.csv')

        if stats:
            logger.debug('meaning the stats')
            mean_stats = mean_df(stats)
            mean_stats.to_csv(model_dir/ 'mean_stats.csv')




def add_to_prefix(
        prompt_prefix,
        icl_file_path,
        num_of_icl_examples,
        all_labels
    ):
    logger.info(f'adding {num_of_icl_examples} to the prompt')
    sample_df = pd.read_csv(icl_file_path)
    logger.info(f'loaded samples from {icl_file_path}')
    if 'reasons' in sample_df.columns:
        ICL_template = ('Input:\n<SAMPLE>\nOutput:\n<REASON>','Input:\n<TEXT>\nOutput:\n')
    else:
        sample_df.labels = sample_df.labels.apply(ast.literal_eval)
        sample_df.labels = sample_df.labels.apply(lambda x: x[0])
        sample_df = sample_df[sample_df.labels != 'OTHER']
        ICL_template = ('Input:\n<SAMPLE>\nOutput:\nThe correct label is <LABEL>\n','Input:\n<TEXT>\nOutput:\n')
    template_text = prompt_prefix + '\n'
    for label in all_labels:
        logger.info(f'adding {num_of_icl_examples} examples for {label}')
        if label != 'OTHER': #don't take any examples from OTHER
            for row in range(0, num_of_icl_examples):
                sample_text = sample_df[sample_df.labels == label].iloc[row]['sentence']
                if 'reasons' in sample_df.columns:
                    sample_reason = sample_df[sample_df.labels == label].iloc[row]['reasons'] #yeah it's plural...
                to_add = ICL_template[0].replace('<SAMPLE>', sample_text)
                if 'reasons' in sample_df.columns:
                    to_add = to_add.replace('<REASON>', sample_reason)
                to_add = to_add.replace('<LABEL>', label)
                template_text += to_add
    template_text += ICL_template[1]
    return template_text



def get_prompt_length_from_config(path, outpath=False):
    hf_model_names = {
        'gemma2:27b': 'google/gemma-2-27b',
        'phi3:medium': 'microsoft/Phi-3-medium-128k-instruct',
        'llama3.1:latest': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'llama3.1:8b-instruct-fp16': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'mistral:7b-instruct-v0.3-fp16' :'mistralai/Mistral-7B-Instruct-v0.3'
    }
    if isinstance(path,str):
        path = Path(path)
    config = read_config(path)
    token_lengths = []
    for model in config['models']:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_names[model])
        for dataset in config['datasets']:
            if dataset.get('add_icl_examples', 0) > 0:
                labels_as_list = dataset.get('label_list', True)
                df = pd.read_csv(dataset['path'])
                if labels_as_list:
                    df[df.columns[dataset['columns']['label']]] = df[df.columns[dataset['columns']['label']]].apply(ast.literal_eval)
                if labels_as_list:
                    all_labels = get_all_labels(df[df.columns[dataset['columns']['label']]])
                else:
                    all_labels = list(df[df.columns[dataset['columns']['label']]].unique()) # casting to list just to be safe!
                template_text = add_to_prefix(
                    dataset['prefix'],
                    dataset['icl_path'],
                    dataset['add_icl_examples'],
                    all_labels
                )
                tokens = tokenizer(template_text)
            else:
                tokens = tokenizer(dataset['prefix'])
            token_lengths.append({
                'model': model,
                'dataset': dataset['name'],
                'token_length': len(tokens.input_ids),
                'num_icl_examples': dataset.get('add_icl_examples', 0)
            })
    token_len_df = pd.DataFrame(token_lengths)
    if outpath:
        token_len_df.to_csv(f'{outpath}/token_lengths.csv')
    print(token_len_df)
    return token_len_df


def generate_icl_examples(file, model, outpath, prefix):
    if isinstance(file,str):
        file = Path(file)
    logger.info(f'creating samples for file')
    sample_df = pd.read_csv(file)
    sample_df.labels = sample_df.labels.apply(ast.literal_eval)
    sample_df.labels = sample_df.labels.apply(lambda x: x[0])
    sample_df = sample_df[sample_df.labels != 'OTHER']
    generation_prompts = [prefix + f'\nText:\n{row["sentence"]}\nLabel:\{row["labels"]}' for _, row in sample_df.iterrows()]
    responses = [generate(generation_prompt, [], model) for generation_prompt in tqdm(generation_prompts,desc='generating')]
    reasons = [' '.join([line for line in response['response'].split('\n') if line and not line.lower().startswith('let me know')]) for response in responses]
    response_fluff_pattern = r'you are(?: absolutely)? right!?\.?'
    reasons = [re.sub(response_fluff_pattern, ' ', reason, flags=re.IGNORECASE)for reason in reasons]
    reasons = [re.sub( r'\s+', ' ', reason, flags=re.IGNORECASE)for reason in reasons]
    sample_df['reasons'] = reasons
    sample_df.to_csv(f'{outpath}/icl_reasons.csv', index=False)


def run_all_experiments(path, overwrite=False):
    if isinstance(path,str):
        path = Path(path)
    config = read_config(path)
    mode = config.get('mode', 'testing')
    for model in config['models']:
        for dataset in config['datasets']:
            for run in range(1, config['number_of_runs'] + 1):

                logger.info(f'test run number {run} for {dataset["name"]} and {model}')
                logger.info(dataset['path'])
                outpath = path / f'{dataset["name"]}/{model}/{run}'
                if (outpath / 'results.csv').exists():
                    if overwrite:
                        logger.info(f'result in {outpath} exists, overwriting!')
                    else:
                        logger.info(f'results in {outpath} already exist, skipping')
                        continue
                outpath.mkdir(exist_ok=True,parents=True)
                if mode == 'testing':
                    test_from_csv(
                        dataset['path'], model, outpath,
                        dataset['prefix'],
                        text_column_index=dataset['columns']['text'],
                        label_column=dataset['columns']['label'], num_of_icl_examples=dataset.get('add_icl_examples', 0),
                        icl_file_path = dataset.get('icl_path',''),
                        labels_as_list=dataset.get('label_list', True)
                    )
                elif mode == 'icl_generation':
                    logger.info(f'generating icl_examples for {model} and {dataset["name"]}')
                    generate_icl_examples(
                        dataset['path'], model, outpath,
                        dataset['prefix'],
                    )


if __name__ == '__main__':
    cli()
