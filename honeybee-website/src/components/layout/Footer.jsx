import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-white text-lg mb-4">HoneyBee</h3>
            <p className="text-slate-600 dark:text-slate-300 mb-4">
              A platform for the development of AI models for oncology with tools for medical data loading,
              embedding generation, and advanced RAG support.
            </p>
          </div>
          
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-white text-lg mb-4">Links</h3>
            <ul className="space-y-2">
              <li>
                <Link 
                  href="https://github.com/lab-rasool/HoneyBee" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-slate-600 hover:text-amber-600 dark:text-slate-300 dark:hover:text-amber-500"
                >
                  GitHub Repository
                </Link>
              </li>
              <li>
                <Link 
                  href="https://arxiv.org/abs/2405.07460" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-slate-600 hover:text-amber-600 dark:text-slate-300 dark:hover:text-amber-500"
                >
                  Research Paper
                </Link>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-white text-lg mb-4">Contact</h3>
            <p className="text-slate-600 dark:text-slate-300">
              For questions about the project, please reach out via GitHub issues.
            </p>
          </div>
        </div>
        
        <div className="mt-10 pt-6 border-t border-slate-200 dark:border-slate-800 text-center text-slate-500 dark:text-slate-400">
          <p>© {new Date().getFullYear()} HoneyBee. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}