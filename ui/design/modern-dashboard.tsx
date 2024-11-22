import React, { useState, useRef } from 'react';
import { ChevronDown, Moon, Sun } from 'lucide-react';

const styles = `
  .app-background {
    min-height: 100vh;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .app-background.dark {
    background: linear-gradient(to bottom right, rgb(8, 8, 13), rgb(13, 13, 18));
  }

  .app-background.light {
    background: linear-gradient(to bottom right, rgb(250, 252, 255), rgb(240, 242, 250));
  }

  .background-shapes {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    opacity: 0.8;
  }

  .background-shape {
    position: absolute;
    border-radius: 50%;
    filter: blur(180px);
    mix-blend-mode: soft-light;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .metric-card {
    backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    isolation: isolate;
    overflow: hidden;
  }

  .metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    z-index: -1;
    transition: opacity 0.5s ease;
  }

  .dark .metric-card {
    background: rgba(15, 15, 20, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.06);
  }

  .dark .metric-card::before {
    background: linear-gradient(
      45deg,
      rgba(255, 255, 255, 0.03) 0%,
      rgba(255, 255, 255, 0.06) 50%,
      rgba(255, 255, 255, 0.03) 100%
    );
  }

  .light .metric-card {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 0, 0, 0.05);
  }

  .light .metric-card::before {
    background: linear-gradient(
      45deg,
      rgba(255, 255, 255, 0.5) 0%,
      rgba(255, 255, 255, 0.8) 50%,
      rgba(255, 255, 255, 0.5) 100%
    );
  }

  .metric-card:hover::before {
    opacity: 0.8;
  }

  .connected-text {
    position: relative;
  }

  .connected-text span {
    position: relative;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    display: inline-block;
    padding: 2px 6px;
    border-radius: 6px;
    margin: 0 1px;
  }

  .connection-container {
    position: absolute;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
  }

  .connection-line {
    position: absolute;
    height: 2px;
    transform-origin: left center;
    z-index: -1;
  }

  .dark .connection-line {
    background: linear-gradient(90deg, 
      rgba(59, 130, 246, 0), 
      rgba(59, 130, 246, 0.3) 50%, 
      rgba(59, 130, 246, 0)
    );
  }

  .light .connection-line {
    background: linear-gradient(90deg, 
      rgba(37, 99, 235, 0), 
      rgba(37, 99, 235, 0.2) 50%, 
      rgba(37, 99, 235, 0)
    );
  }

  .dark .connected-text span.active {
    color: rgb(255, 255, 255);
    background: rgba(59, 130, 246, 0.2);
    font-weight: 500;
    transform: scale(1.05);
  }

  .light .connected-text span.active {
    color: rgb(0, 0, 0);
    background: rgba(37, 99, 235, 0.1);
    font-weight: 500;
    transform: scale(1.05);
  }

  .dark .connected-text span.related-1 {
    color: rgba(255, 255, 255, 0.95);
    background: rgba(59, 130, 246, 0.15);
  }

  .dark .connected-text span.related-2 {
    color: rgba(255, 255, 255, 0.85);
    background: rgba(59, 130, 246, 0.1);
  }

  .dark .connected-text span.related-3 {
    color: rgba(255, 255, 255, 0.75);
    background: rgba(59, 130, 246, 0.05);
  }

  .light .connected-text span.related-1 {
    color: rgba(0, 0, 0, 0.95);
    background: rgba(37, 99, 235, 0.08);
  }

  .light .connected-text span.related-2 {
    color: rgba(0, 0, 0, 0.85);
    background: rgba(37, 99, 235, 0.06);
  }

  .light .connected-text span.related-3 {
    color: rgba(0, 0, 0, 0.75);
    background: rgba(37, 99, 235, 0.04);
  }
`;

const BackgroundShapes = ({ isDark }) => (
  <div className="background-shapes">
    <div 
      className="background-shape"
      style={{
        width: '140vw',
        height: '140vw',
        background: isDark 
          ? 'radial-gradient(circle, rgba(255, 255, 255, 0.03) 0%, transparent 70%)'
          : 'radial-gradient(circle, rgba(0, 0, 0, 0.02) 0%, transparent 70%)',
        top: '-20vh',
        left: '50%',
        transform: 'translateX(-50%)',
      }}
    />
    <div 
      className="background-shape"
      style={{
        width: '100vw',
        height: '100vw',
        background: isDark
          ? 'radial-gradient(circle, rgba(255, 255, 255, 0.02) 0%, transparent 70%)'
          : 'radial-gradient(circle, rgba(0, 0, 0, 0.015) 0%, transparent 70%)',
        bottom: '-30vh',
        right: '-20vw',
      }}
    />
  </div>
);

const MetricCard = ({ title, value, change, isDark }) => {
  const isPositive = change.startsWith('+');
  const changeColor = isPositive ? 'text-emerald-400' : 'text-red-400';

  return (
    <div className="metric-card rounded-xl p-6 transition-all duration-300">
      <div className="space-y-2">
        <div className={`text-sm font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'} mb-3`}>{title}</div>
        <div className={`text-4xl font-light tracking-tight mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>{value}</div>
        <div className={`text-sm flex items-center gap-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          <div className="w-full h-1 rounded-full bg-gray-800 overflow-hidden">
            <div 
              className={`h-full rounded-full ${isPositive ? 'bg-emerald-500/20' : 'bg-red-500/20'} transition-all duration-500`} 
              style={{ width: `${Math.min(Math.abs(parseFloat(change)) * 10, 100)}%` }}
            />
          </div>
          <span className={`font-medium min-w-[3.5rem] text-right ${isDark ? changeColor : ''}`}>{change}</span>
        </div>
      </div>
    </div>
  );
};

const WordCloud = ({ isDark }) => {
  const words = "Artificial Intelligence is reshaping the landscape of modern technology creating unprecedented opportunities for innovation and discovery".split(" ");
  const [activeWord, setActiveWord] = useState(null);
  const [relatedWords, setRelatedWords] = useState({});
  const [connections, setConnections] = useState([]);
  const wordRefs = useRef([]);

  const createConnection = (start, end) => {
    const startRect = wordRefs.current[start].getBoundingClientRect();
    const endRect = wordRefs.current[end].getBoundingClientRect();
    const containerRect = wordRefs.current[start].parentElement.getBoundingClientRect();

    const x1 = startRect.left + startRect.width / 2 - containerRect.left;
    const y1 = startRect.top + startRect.height / 2 - containerRect.top;
    const x2 = endRect.left + endRect.width / 2 - containerRect.left;
    const y2 = endRect.top + endRect.height / 2 - containerRect.top;

    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

    return {
      left: x1,
      top: y1,
      width: length,
      transform: `rotate(${angle}deg)`,
      opacity: Math.max(0.1, 0.5 - (length / 1000))
    };
  };

  const handleWordHover = (index) => {
    if (index === activeWord) return;

    const numRelated = Math.floor(Math.random() * 3) + 2;
    const related = {};
    const wordIndices = [...Array(words.length).keys()];
    const newConnections = [];
    
    wordIndices.splice(index, 1);
    
    for (let i = 1; i <= numRelated; i++) {
      if (wordIndices.length === 0) break;
      const randomIndex = Math.floor(Math.random() * wordIndices.length);
      const selectedIndex = wordIndices[randomIndex];
      related[selectedIndex] = i;
      newConnections.push(createConnection(index, selectedIndex));
      wordIndices.splice(randomIndex, 1);
    }

    setActiveWord(index);
    setRelatedWords(related);
    setConnections(newConnections);
  };

  const getWordClass = (index) => {
    if (index === activeWord) return 'active';
    if (relatedWords[index]) return `related-${relatedWords[index]}`;
    return '';
  };

  return (
    <div className={`connected-text text-lg leading-relaxed ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
      <div className="connection-container">
        {connections.map((style, index) => (
          <div
            key={index}
            className="connection-line"
            style={style}
          />
        ))}
      </div>
      {words.map((word, index) => (
        <span
          key={index}
          ref={el => wordRefs.current[index] = el}
          className={getWordClass(index)}
          onMouseEnter={() => handleWordHover(index)}
          onMouseLeave={() => {
            setActiveWord(null);
            setRelatedWords({});
            setConnections([]);
          }}
        >
          {word}
        </span>
      ))}
    </div>
  );
};

const ThemeToggle = ({ isDark, onToggle }) => (
  <button 
    onClick={onToggle}
    className={`fixed top-6 right-6 z-10 p-3 rounded-xl transition-all hover:scale-110 ${
      isDark 
        ? 'bg-gray-800/50 text-gray-200 hover:bg-gray-700/50' 
        : 'bg-white/50 text-gray-800 hover:bg-gray-50/50'
    } backdrop-blur-md`}
  >
    {isDark ? <Sun size={20} /> : <Moon size={20} />}
  </button>
);

const AnalyticsDashboard = () => {
  const [isDark, setIsDark] = useState(true);

  return (
    <>
      <style>{styles}</style>
      <div className={`app-background ${isDark ? 'dark' : 'light'}`}>
        <BackgroundShapes isDark={isDark} />
        <ThemeToggle isDark={isDark} onToggle={() => setIsDark(!isDark)} />
        
        <div className="max-w-6xl mx-auto pt-24 px-6 relative">
          <div className="mb-16">
            <h1 className={`text-5xl font-light mb-3 tracking-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Welcome back, Adam
            </h1>
            <p className={`text-lg ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Track performance and user engagement metrics
            </p>
          </div>

          <div className="grid grid-cols-3 gap-6 mb-12">
            <MetricCard 
              isDark={isDark}
              title="Total Reach"
              value="2,340,560"
              change="+8%"
            />
            <MetricCard 
              isDark={isDark}
              title="Active Campaigns"
              value="15"
              change="+10%"
            />
            <MetricCard 
              isDark={isDark}
              title="Engagement Rate"
              value="14.3%"
              change="+5%"
            />
          </div>

          <div className="grid grid-cols-2 gap-6 pb-12">
            <div className="metric-card rounded-xl p-8">
              <h3 className={`text-lg mb-8 flex items-center justify-between ${
                isDark ? 'text-white' : 'text-gray-900'
              }`}>
                <span className="font-medium">Word Connections</span>
                <div className="text-sm opacity-60 flex items-center gap-2 cursor-pointer hover:opacity-80">
                  Weekly <ChevronDown size={16} />
                </div>
              </h3>
              <WordCloud isDark={isDark} />
            </div>

            <div className="metric-card rounded-xl p-8">
              <h3 className={`text-lg mb-8 font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Activity Heatmap
              </h3>
              <div className="grid grid-cols-7 gap-2">
                {Array.from({ length: 35 }).map((_, i) => {
                  const intensity = Math.random();
                  return (
                    <div
                      key={i}
                      className="h-12 rounded-lg transition-all duration-300 hover:scale-105 cursor-pointer"
                      style={{
                        background: isDark
                          ? `rgba(59, 130, 246, ${intensity * 0.3})`
                          : `rgba(37, 99, 235, ${intensity * 0.2})`
                      }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AnalyticsDashboard;