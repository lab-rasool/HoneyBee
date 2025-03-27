import { motion } from 'framer-motion';
import clsx from 'clsx';

export default function Card({ title, children, icon, className, hoverEffect = true }) {
  return (
    <motion.div
      whileHover={hoverEffect ? { y: -5 } : undefined}
      className={clsx(
        "bg-white dark:bg-slate-800 rounded-xl shadow-md overflow-hidden border border-slate-200 dark:border-slate-700",
        className
      )}
    >
      <div className="p-6">
        {icon && (
          <div className="mb-4 inline-flex p-3 rounded-lg bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-500">
            {icon}
          </div>
        )}
        {title && <h3 className="text-xl font-semibold mb-2 text-slate-900 dark:text-white">{title}</h3>}
        <div className="text-slate-600 dark:text-slate-300">{children}</div>
      </div>
    </motion.div>
  );
}