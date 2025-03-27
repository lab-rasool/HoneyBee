'use client';

import { useState } from 'react';
import { CheckIcon, DocumentDuplicateIcon } from '@heroicons/react/24/outline';

export default function CitationCard() {
  const [copied, setCopied] = useState('');
  
  const citationData = {
    bibtex: `@article{honeybee,
      title={HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models}, 
      author={Aakash Tripathi and Asim Waqas and Yasin Yilmaz and Ghulam Rasool},
      year={2024},
      eprint={2405.07460},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}`,
    plainText: "Tripathi, A., Waqas, A., Yilmaz, Y., & Rasool, G. (2024). HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets with Foundational Embedding Models. arXiv preprint arXiv:2405.07460."
  };

  const copyToClipboard = (text, type) => {
    navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(''), 2000);
  };

  return (
    <section className="py-16 bg-amber-50">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900">Citation</h2>
          <p className="mt-4 text-lg text-gray-600">
            If you use HoneyBee in your research, please cite our paper:
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-amber-600">HoneyBee Paper</h3>
              <a
                href="https://arxiv.org/abs/2405.07460"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amber-600 hover:text-amber-800 font-medium"
              >
                View on arXiv
              </a>
            </div>
            
            <div className="mt-4 flex flex-col gap-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-sm font-medium text-gray-500">BibTeX</h4>
                  <button
                    onClick={() => copyToClipboard(citationData.bibtex, 'bibtex')}
                    className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded text-amber-700 hover:bg-amber-100"
                  >
                    {copied === 'bibtex' ? (
                      <>
                        <CheckIcon className="h-3 w-3 mr-1" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <DocumentDuplicateIcon className="h-3 w-3 mr-1" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <pre className="bg-gray-50 rounded-md p-3 text-xs overflow-x-auto">
                  {citationData.bibtex}
                </pre>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-sm font-medium text-gray-500">Plain Text</h4>
                  <button
                    onClick={() => copyToClipboard(citationData.plainText, 'plain')}
                    className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium rounded text-amber-700 hover:bg-amber-100"
                  >
                    {copied === 'plain' ? (
                      <>
                        <CheckIcon className="h-3 w-3 mr-1" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <DocumentDuplicateIcon className="h-3 w-3 mr-1" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <div className="bg-gray-50 rounded-md p-3 text-sm">
                  {citationData.plainText}
                </div>
              </div>
            </div>

            <div className="mt-6 border-t border-gray-200 pt-4">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <img 
                    src="/assets/honeybee-icon.png" 
                    alt="HoneyBee Logo" 
                    className="h-8 w-8"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = "/assets/fallback-icon.png";
                    }}
                  />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">
                    HoneyBee: A Scalable Modular Framework for Creating Multimodal Oncology Datasets
                  </p>
                  <p className="text-xs text-gray-500">
                    Aakash Tripathi, Asim Waqas, Yasin Yilmaz, Ghulam Rasool • 2024
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}