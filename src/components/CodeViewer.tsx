import React, { useState, useEffect } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import { Copy, Check, Download, FileCode, Terminal, Sparkles } from 'lucide-react';

interface CodeViewerProps {
  code: string;
  filename: string;
  language?: string;
  title?: string;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({
  code,
  filename,
  language = 'python',
  title,
}) => {
  const [copied, setCopied] = useState(false);
  const [highlightedCode, setHighlightedCode] = useState('');

  useEffect(() => {
    try {
      const grammar = Prism.languages[language] || Prism.languages.python;
      const html = Prism.highlight(code, grammar, language);
      setHighlightedCode(html);
    } catch (e) {
      setHighlightedCode(code);
    }
  }, [code, language]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const lineCount = code.split('\n').length;

  return (
    <div className="flex flex-col border-2 border-white bg-black shadow-[6px_6px_0px_0px_rgba(255,255,255,1)]">
      {/* Code Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between px-4 py-3 bg-zinc-950 border-b-2 border-white gap-3 select-none">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 bg-yellow-400 text-black font-black flex items-center justify-center border border-white">
            <FileCode className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-black text-sm uppercase tracking-wider text-white">
                {filename}
              </span>
              <span className="font-mono text-[10px] font-black uppercase bg-white text-black px-1.5 py-0.2 border border-white">
                {lineCount} LINES
              </span>
            </div>
            {title && (
              <span className="font-mono text-[10px] text-zinc-400 uppercase tracking-widest hidden sm:inline">
                {title}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-black hover:bg-white hover:text-black text-white border-2 border-white transition font-mono text-xs font-black uppercase tracking-wider cursor-pointer shadow-[2px_2px_0px_0px_rgba(255,255,255,1)] active:translate-x-0.5 active:translate-y-0.5"
            title="Copy to clipboard"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 stroke-[3] text-yellow-400" />
                <span className="text-yellow-400">COPIED!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 stroke-[2.5]" />
                <span>COPY CODE</span>
              </>
            )}
          </button>

          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-yellow-400 hover:bg-yellow-300 text-black border-2 border-white transition font-mono text-xs font-black uppercase tracking-wider cursor-pointer shadow-[2px_2px_0px_0px_rgba(255,255,255,1)] active:translate-x-0.5 active:translate-y-0.5"
            title="Download script"
          >
            <Download className="w-3.5 h-3.5 stroke-[3]" />
            <span>EXPORT .PY</span>
          </button>
        </div>
      </div>

      {/* Code Editor Body */}
      <div className="relative font-mono text-[13px] leading-relaxed bg-[#050505] text-zinc-100 overflow-x-auto max-h-[640px] p-5">
        <pre className="m-0 p-0 overflow-visible">
          <code
            className={`language-${language}`}
            dangerouslySetInnerHTML={{ __html: highlightedCode || code }}
          />
        </pre>
      </div>

      {/* Terminal Quick Run Hint Footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 bg-zinc-950 border-t-2 border-white text-xs font-mono">
        <div className="flex items-center gap-2 text-zinc-300">
          <Terminal className="w-4 h-4 text-yellow-400 stroke-[2.5]" />
          <span className="font-bold uppercase tracking-wider text-zinc-400">RUN:</span>
          <code className="text-white font-black bg-black border border-zinc-700 px-2 py-0.5">
            python {filename}
          </code>
        </div>
        <div className="flex items-center gap-2 text-zinc-400 uppercase font-bold text-[11px] tracking-wider">
          <Sparkles className="w-3.5 h-3.5 text-yellow-400" />
          <span>PRODUCTION-READY PIPELINE // CUDA ACCELERATED</span>
        </div>
      </div>
    </div>
  );
};

