import { useState } from 'react';
import Button from '../ui/Button';
import { motion } from 'framer-motion';
import { CheckIcon, ClipboardIcon } from '@heroicons/react/24/outline';

export default function CitationSection() {
  const [copied, setCopied] = useState(false);
  
  const citationText = `@article{honeybee,
      title={HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models}, 
      author={Aakash Tripathi and Asim Waqas and Yasin Yilmaz and Ghulam Rasool},
      year={2024},
      eprint={2405.07460},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(citationText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="py-16 bg-white dark:bg-slate-900">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white sm:text-4xl">Citation</h2>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-slate-600 dark:text-slate-300">
            If you use HoneyBee in your research, please cite our paper
          </p>
        </div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-3xl mx-auto bg-slate-50 dark:bg-slate-800 rounded-xl overflow-hidden shadow-md border border-slate-200 dark:border-slate-700"
        >
          <div className="flex justify-between items-center p-4 bg-slate-100 dark:bg-slate-700">
            <h3 className="font-medium text-slate-700 dark:text-slate-200">BibTeX Citation</h3>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={copyToClipboard}
              className="flex gap-2 items-center"
            >
              {copied ? <CheckIcon className="h-4 w-4" /> : <ClipboardIcon className="h-4 w-4" />}
              {copied ? 'Copied!' : 'Copy'}
            </Button>
          </div>
          <pre className="p-6 overflow-x-auto text-sm text-slate-800 dark:text-slate-200 font-mono">
            {citationText}
          </pre>
        </motion.div>
      </div>
    </section>
  );
}