#!/usr/bin/env python3

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import logging
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model to evaluate
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use for inference
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = self._setup_device(device)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Evaluation data
        self.test_datasets = {}
        self.results = {}
        
        # Load model
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                logger.info(f"Using GPU: {device}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        return device
        
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if "cuda" in self.device else None,
                trust_remote_code=True
            )
            
            if "cpu" in self.device:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if "cuda" in self.device else None,
                torch_dtype=torch.float16
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_test_data(self, data_path: str, data_type: str = "auto"):
        """
        Load test dataset.
        
        Args:
            data_path: Path to test data file
            data_type: Type of data ('json', 'jsonl', 'csv', 'auto')
        """
        try:
            if data_type == "auto":
                data_type = Path(data_path).suffix[1:]
            
            if data_type == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_type == "jsonl":
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            elif data_type == "csv":
                df = pd.read_csv(data_path)
                data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            dataset_name = Path(data_path).stem
            self.test_datasets[dataset_name] = data
            logger.info(f"Loaded {len(data)} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, 
                         temperature: float = 0.7, top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated response
        """
        try:
            with torch.no_grad():
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = outputs[0]['generated_text'].strip()
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_qa_accuracy(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate question-answering accuracy.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            
        Returns:
            Evaluation results
        """
        if dataset_name not in self.test_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Evaluating QA accuracy on {dataset_name}")
        
        dataset = self.test_datasets[dataset_name]
        correct = 0
        total = 0
        predictions = []
        ground_truths = []
        response_times = []
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            # Extract question and answer
            question = sample.get('question', sample.get('instruction', ''))
            expected_answer = sample.get('answer', sample.get('output', ''))
            
            if not question or not expected_answer:
                continue
            
            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(prompt, max_new_tokens=128, temperature=0.1)
            response_time = time.time() - start_time
            
            predictions.append(response)
            ground_truths.append(expected_answer)
            response_times.append(response_time)
            
            # Simple accuracy check (can be enhanced with semantic similarity)
            if self._check_answer_similarity(response, expected_answer):
                correct += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_response_time = np.mean(response_times)
        
        results = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_response_time': avg_response_time,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'response_times': response_times
        }
        
        self.results[f"{dataset_name}_qa"] = results
        logger.info(f"QA Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return results
    
    def _check_answer_similarity(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        Uses simple keyword matching - can be enhanced with semantic similarity.
        """
        pred_clean = prediction.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        # Exact match
        if pred_clean == truth_clean:
            return True
        
        # Keyword overlap
        pred_words = set(pred_clean.split())
        truth_words = set(truth_clean.split())
        
        if len(truth_words) == 0:
            return False
        
        overlap = len(pred_words.intersection(truth_words))
        similarity = overlap / len(truth_words)
        
        return similarity > 0.7  # 70% keyword overlap threshold
    
    def evaluate_response_quality(self, dataset_name: str, sample_size: int = 50) -> Dict[str, Any]:
        """
        Evaluate response quality using multiple metrics.
        
        Args:
            dataset_name: Name of the dataset
            sample_size: Number of samples to evaluate
            
        Returns:
            Quality evaluation results
        """
        if dataset_name not in self.test_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        logger.info(f"Evaluating response quality on {dataset_name}")
        
        dataset = self.test_datasets[dataset_name][:sample_size]
        
        quality_scores = {
            'coherence': [],
            'relevance': [],
            'completeness': [],
            'fluency': []
        }
        
        for i, sample in enumerate(dataset):
            question = sample.get('question', sample.get('instruction', ''))
            if not question:
                continue
            
            response = self.generate_response(f"Question: {question}\nAnswer:")
            
            # Calculate quality metrics
            coherence = self._calculate_coherence(response)
            relevance = self._calculate_relevance(question, response)
            completeness = self._calculate_completeness(response)
            fluency = self._calculate_fluency(response)
            
            quality_scores['coherence'].append(coherence)
            quality_scores['relevance'].append(relevance)
            quality_scores['completeness'].append(completeness)
            quality_scores['fluency'].append(fluency)
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(dataset)} samples")
        
        avg_scores = {metric: np.mean(scores) for metric, scores in quality_scores.items()}
        
        results = {
            'dataset': dataset_name,
            'sample_size': len(dataset),
            'avg_scores': avg_scores,
            'detailed_scores': quality_scores
        }
        
        self.results[f"{dataset_name}_quality"] = results
        logger.info(f"Quality scores: {avg_scores}")
        
        return results
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Simple coherence metric based on sentence length consistency
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        avg_length = np.mean(lengths)
        variance = np.var(lengths)
        
        # Lower variance indicates better coherence
        coherence = max(0, 1 - (variance / (avg_length + 1)))
        return min(1, coherence)
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate answer relevance to question."""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        q_words -= stop_words
        a_words -= stop_words
        
        if not q_words:
            return 0.5
        
        overlap = len(q_words.intersection(a_words))
        return overlap / len(q_words)
    
    def _calculate_completeness(self, text: str) -> float:
        """Calculate response completeness."""
        words = text.split()
        
        # Length-based completeness
        if len(words) < 5:
            return 0.2
        elif len(words) < 20:
            return 0.6
        elif len(words) < 100:
            return 1.0
        else:
            return 0.8  # Very long responses might be repetitive
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate text fluency."""
        # Simple fluency metric based on sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check for basic sentence structure
        fluency_score = 0
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 3:  # Minimum sentence length
                fluency_score += 1
        
        return fluency_score / len(sentences)
    
    def compare_models(self, baseline_model_path: str, dataset_name: str, 
                      test_prompts: List[str] = None) -> Dict[str, Any]:
        """
        Compare current model with a baseline model.
        
        Args:
            baseline_model_path: Path to baseline model
            dataset_name: Dataset for comparison
            test_prompts: Optional list of specific prompts to test
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing with baseline model: {baseline_model_path}")
        
        # Load baseline model
        baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.float16,
            device_map="auto" if "cuda" in self.device else None
        )
        
        baseline_pipeline = pipeline(
            "text-generation",
            model=baseline_model,
            tokenizer=baseline_tokenizer,
            device_map="auto" if "cuda" in self.device else None,
            torch_dtype=torch.float16
        )
        
        if test_prompts is None:
            # Use samples from dataset
            if dataset_name in self.test_datasets:
                dataset = self.test_datasets[dataset_name][:20]  # Test on 20 samples
                test_prompts = [
                    sample.get('question', sample.get('instruction', ''))
                    for sample in dataset
                    if sample.get('question', sample.get('instruction', ''))
                ]
            else:
                test_prompts = [
                    "What is artificial intelligence?",
                    "Explain the concept of machine learning.",
                    "How does deep learning work?",
                    "What are the benefits of renewable energy?",
                    "Describe the process of photosynthesis."
                ]
        
        comparison_results = []
        
        for prompt in test_prompts:
            # Generate with current model
            current_response = self.generate_response(f"Question: {prompt}\nAnswer:")
            
            # Generate with baseline
            baseline_outputs = baseline_pipeline(
                f"Question: {prompt}\nAnswer:",
                max_new_tokens=128,
                temperature=0.1,
                return_full_text=False
            )
            baseline_response = baseline_outputs[0]['generated_text'].strip()
            
            comparison_results.append({
                'prompt': prompt,
                'current_model': current_response,
                'baseline_model': baseline_response,
                'current_quality': np.mean([
                    self._calculate_coherence(current_response),
                    self._calculate_relevance(prompt, current_response),
                    self._calculate_completeness(current_response),
                    self._calculate_fluency(current_response)
                ]),
                'baseline_quality': np.mean([
                    self._calculate_coherence(baseline_response),
                    self._calculate_relevance(prompt, baseline_response),
                    self._calculate_completeness(baseline_response),
                    self._calculate_fluency(baseline_response)
                ])
            })
        
        results = {
            'baseline_model': baseline_model_path,
            'test_prompts': len(test_prompts),
            'comparisons': comparison_results,
            'avg_current_quality': np.mean([r['current_quality'] for r in comparison_results]),
            'avg_baseline_quality': np.mean([r['baseline_quality'] for r in comparison_results])
        }
        
        self.results['model_comparison'] = results
        logger.info(f"Comparison complete. Current: {results['avg_current_quality']:.3f}, Baseline: {results['avg_baseline_quality']:.3f}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode."""
        print("\n" + "="*60)
        print("INTERACTIVE MODEL TESTING")
        print("="*60)
        print("Enter questions to test your model.")
        print("Commands: 'quit', 'exit', 'q' to exit")
        print("Commands: 'stats' to show session statistics")
        print("-"*60)
        
        session_stats = {
            'questions_asked': 0,
            'avg_response_time': 0,
            'total_time': 0
        }
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'stats':
                    self._print_session_stats(session_stats)
                    continue
                elif not question:
                    continue
                
                print(f"\n🤖 Model thinking...")
                start_time = time.time()
                
                response = self.generate_response(f"Question: {question}\nAnswer:")
                
                response_time = time.time() - start_time
                
                print(f"\n💡 Answer: {response}")
                print(f"⏱️  Response time: {response_time:.2f} seconds")
                
                # Update stats
                session_stats['questions_asked'] += 1
                session_stats['total_time'] += response_time
                session_stats['avg_response_time'] = session_stats['total_time'] / session_stats['questions_asked']
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        self._print_session_stats(session_stats)
    
    def _print_session_stats(self, stats: Dict[str, Any]):
        """Print session statistics."""
        print("\n" + "-"*40)
        print("SESSION STATISTICS")
        print("-"*40)
        print(f"Questions asked: {stats['questions_asked']}")
        print(f"Average response time: {stats['avg_response_time']:.2f}s")
        print(f"Total testing time: {stats['total_time']:.2f}s")
        print("-"*40)
    
    def benchmark_performance(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            num_samples: Number of samples for benchmarking
            
        Returns:
            Performance metrics
        """
        logger.info(f"Running performance benchmark with {num_samples} samples")
        
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do you make chocolate chip cookies?",
            "What are the benefits of exercise?",
            "Describe the water cycle."
        ]
        
        response_times = []
        token_counts = []
        
        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]
            
            start_time = time.time()
            response = self.generate_response(f"Question: {prompt}\nAnswer:")
            response_time = time.time() - start_time
            
            response_times.append(response_time)
            token_counts.append(len(self.tokenizer.encode(response)))
            
            if i % 20 == 0:
                logger.info(f"Benchmark progress: {i+1}/{num_samples}")
        
        results = {
            'num_samples': num_samples,
            'avg_response_time': np.mean(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'std_response_time': np.std(response_times),
            'avg_tokens_generated': np.mean(token_counts),
            'tokens_per_second': np.mean(token_counts) / np.mean(response_times)
        }
        
        self.results['performance_benchmark'] = results
        logger.info(f"Benchmark complete. Avg time: {results['avg_response_time']:.2f}s, Tokens/sec: {results['tokens_per_second']:.1f}")
        
        return results
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.json"
        
        # Compile all results
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': self.device,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_report_summary(report)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        summary = {
            'evaluations_run': len(self.results),
            'datasets_tested': len(self.test_datasets)
        }
        
        # Add specific metrics if available
        for key, result in self.results.items():
            if 'accuracy' in result:
                summary[f'{key}_accuracy'] = result['accuracy']
            if 'avg_scores' in result:
                summary[f'{key}_avg_quality'] = np.mean(list(result['avg_scores'].values()))
        
        return summary
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """Print evaluation report summary."""
        print("\n" + "="*70)
        print("MODEL EVALUATION REPORT SUMMARY")
        print("="*70)
        print(f"Model: {report['model_path']}")
        print(f"Evaluation Time: {report['evaluation_timestamp']}")
        print(f"Device: {report['device']}")
        print("-"*70)
        
        summary = report['summary']
        print(f"Evaluations Run: {summary['evaluations_run']}")
        print(f"Datasets Tested: {summary['datasets_tested']}")
        
        # Print key metrics
        for key, value in summary.items():
            if key not in ['evaluations_run', 'datasets_tested']:
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        print("="*70)
    
    def visualize_results(self, save_plots: bool = True):
        """Generate visualization plots for evaluation results."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation Results - {Path(self.model_path).name}', fontsize=16)
        
        # Plot 1: Accuracy scores
        ax1 = axes[0, 0]
        accuracies = []
        datasets = []
        for key, result in self.results.items():
            if 'accuracy' in result:
                accuracies.append(result['accuracy'])
                datasets.append(key.replace('_qa', ''))
        
        if accuracies:
            bars = ax1.bar(datasets, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
            ax1.set_title('Accuracy by Dataset')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # Plot 2: Response times
        ax2 = axes[0, 1]
        if 'performance_benchmark' in self.results:
            perf = self.results['performance_benchmark']
            metrics = ['avg_response_time', 'min_response_time', 'max_response_time']
            values = [perf[m] for m in metrics]
            labels = ['Average', 'Minimum', 'Maximum']
            
            bars = ax2.bar(labels, values, color=['orange', 'green', 'red'], alpha=0.7)
            ax2.set_title('Response Time Analysis')
            ax2.set_ylabel('Time (seconds)')
            
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}s', ha='center', va='bottom')
        
        # Plot 3: Quality scores
        ax3 = axes[1, 0]
        quality_data = []
        for key, result in self.results.items():
            if 'avg_scores' in result:
                quality_data.append(result['avg_scores'])
        
        if quality_data:
            # Average across all datasets
            avg_quality = {}
            for metric in quality_data[0].keys():
                avg_quality[metric] = np.mean([d[metric] for d in quality_data])
            
            metrics = list(avg_quality.keys())
            values = list(avg_quality.values())
            
            bars = ax3.bar(metrics, values, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            ax3.set_title('Average Quality Scores')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 1)
            
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # Plot 4: Model comparison (if available)
        ax4 = axes[1, 1]
        if 'model_comparison' in self.results:
            comp = self.results['model_comparison']
            models = ['Current Model', 'Baseline Model']
            scores = [comp['avg_current_quality'], comp['avg_baseline_quality']]
            
            bars = ax4.bar(models, scores, color=['blue', 'gray'], alpha=0.7)
            ax4.set_title('Model Comparison')
            ax4.set_ylabel('Quality Score')
            ax4.set_ylim(0, 1)
            
            for bar, score in zip(bars, scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"evaluation_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {plot_path}")
        
        plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluator")
    parser.add_argument("model_path", help="Path to the model to evaluate")
    parser.add_argument("--tokenizer", help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--test-data", help="Path to test dataset")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run performance benchmark")
    parser.add_argument("--compare", help="Path to baseline model for comparison")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--output", "-o", help="Output path for evaluation report")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer,
        device=args.device
    )
    
    # Load test data if provided
    if args.test_data:
        evaluator.load_test_data(args.test_data)
    
    # Run evaluations
    if args.interactive:
        evaluator.interactive_test()
    
    if args.test_data:
        dataset_name = Path(args.test_data).stem
        evaluator.evaluate_qa_accuracy(dataset_name)
        evaluator.evaluate_response_quality(dataset_name)
    
    if args.benchmark:
        evaluator.benchmark_performance()
    
    if args.compare:
        dataset_name = Path(args.test_data).stem if args.test_data else "default"
        evaluator.compare_models(args.compare, dataset_name)
    
    # Generate report
    report_path = evaluator.generate_report(args.output)
    
    # Generate plots
    if args.visualize:
        evaluator.visualize_results()
    
    print(f"\nEvaluation complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
