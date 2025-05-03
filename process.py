import random
import argparse
import logging

class Process:
    '''
    Template class for processes.
    A process is simply put running the specified tasks associated with the specified model and dataset.
    
    Create a child class and implement the __init__ method with the following attributes:
    dm_ids: List of dataset ids
    tasks: Dictionary of tasks, where the keys are dataset ids and the values are dictionaries of tasks, where the keys are model ids and the values are TranslationTask objects
    model_ids: List of model ids
    
    The argument parsing and fancy prints is handled by this script to make it easy to create processes
    '''
    def __init__(self):

        self.dm_ids = None
        self.tasks = None
        self.model_ids = None

    def show_task_detailed(self, model=None, dataset=None):
        if model is None:
            model = random.choice(self.model_ids)
        if dataset is None:
            dataset = random.choice(self.dm_ids)

        print(f"Task details for {dataset} - {model}:")
        task_vars = vars(self.tasks[dataset][model])
        for key, value in task_vars.items():
            print(f"  {key}: {value}")

    def show_options(self):
        print('Available datasets:', self.dm_ids)
        print('Available models:', self.model_ids)

    def check_model_dataset(self, model, dataset):
        '''Check if model and dataset are valid'''
        if dataset not in self.dm_ids:
            print(f'Error: Invalid dataset. Must be one of {self.dm_ids}')
            return False

        if model not in self.model_ids:
            print(f'Error: Invalid model. Must be one of {self.model_ids}')
            return False
        return True


def show_commands():
    print('Available commands:')
    print('options: Show available datasets and models')
    print('task: Show detailed information about a task')
    print(
        'task -model $model -dataset $dataset: Show detailed information about a task')
    print('run -model $model -dataset $dataset: Run a task')


def proc_parser(desc='Process Template'):
    parser = argparse.ArgumentParser(
        description=desc)
    subparsers = parser.add_subparsers(
        dest='command', help='Command to execute')

    subparsers.add_parser(
        'options', help='Show available options')

    task_parser = subparsers.add_parser(
        'task', help='Show task details (if no arguments, show random task details)')
    task_parser.add_argument('-model', '-m', type=str, help='Model name')
    task_parser.add_argument('-dataset', '-d', type=str, help='Dataset name')

    run_parser = subparsers.add_parser('run', help='Run a specific task')
    run_parser.add_argument(
        '-model', '-m', type=str, required=True, help='Model name')
    run_parser.add_argument('-dataset',  '-d', type=str,
                            required=True, help='Dataset name')
    return parser


def main(parser=proc_parser(), proc=Process()):
    args = parser.parse_args()
    if args.command is None:
        show_commands()
        return

    if args.command == 'options':
        proc.show_options()
        return
    elif args.command == 'task':
        if args.model is None and args.dataset is None:
            proc.show_task_detailed()
        else:
            if proc.check_model_dataset(args.model, args.dataset):
                proc.show_task_detailed(model=args.model, dataset=args.dataset)
        return
    elif args.command == 'run':
        if proc.check_model_dataset(args.model, args.dataset):
            logging.info(
                f"[🚀]: Selected task for {args.dataset} - {args.model}")
            proc.tasks[args.dataset][args.model].run()
            return
    else:
        parser.print_help()
        return


if __name__ == '__main__':
    print('This is a template to make it easier to implement processes.')
    print('Please implement the Process class and run the main function.')
    print('Or run the phase0.py file.')
