import React, { useState } from 'react';
import { Sun, Moon, MessageCircle, X, Send } from 'lucide-react';

type Message = {
  text: string;
  isBot: boolean;
};

export default function App() {
  const [isDark, setIsDark] = useState(true);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { text: "Hello! How can I help you today?", isBot: true }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const toggleTheme = () => setIsDark(!isDark);
  const handleChatbotClick = () => setIsChatOpen(!isChatOpen);

  // 🔥 MAIN CHANGE: real backend call
  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = inputValue;

    // Add user message
    setMessages((prev: Message[]) => [...prev, { text: userMessage, isBot: false }]);
    setInputValue('');
    setLoading(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();

      setMessages((prev: Message[]) => [
        ...prev,
        { text: data.response || data.answer || "No response from server.", isBot: true }
      ]);

    } catch (err: any) {
      let errorMessage = "⚠️ Server not reachable. Please try again.";
      
      if (err.name === 'AbortError') {
        errorMessage = "⚠️ Request timed out. The server is taking too long to respond.";
      } else if (err instanceof TypeError) {
        errorMessage = "⚠️ Connection failed. Make sure the backend server is running on port 8000.";
      }
      
      setMessages((prev: Message[]) => [
        ...prev,
        { text: errorMessage, isBot: true }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSendMessage();
  };

  /* ---- STYLES (unchanged except minor safety) ---- */
  const styles = {
    container: {
      width: '100%',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'space-between',
      backgroundColor: isDark ? '#000000' : '#ffffff',
      overflow: 'hidden' as const,
    },
    header: {
      width: '100%',
      display: 'flex',
      justifyContent: 'flex-end',
      padding: '16px 24px',
      flexShrink: 0,
    },
    themeButton: {
      padding: '8px',
      borderRadius: '50%',
      border: 'none',
      cursor: 'pointer',
      backgroundColor: isDark ? '#1f2937' : '#e5e7eb',
      color: isDark ? '#fbbf24' : '#1f2937',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    centerContent: {
      flex: 1,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '40px 20px',
    },
    heading: {
      fontSize: 'clamp(2rem, 5vw, 3.5rem)',
      textAlign: 'center' as const,
      color: isDark ? '#ffffff' : '#000000',
      margin: 0,
    },
    bottomSection: { 
      paddingBottom: '24px',
      flexShrink: 0,
    },
    chatButton: {
      width: '56px',
      height: '56px',
      borderRadius: '50%',
      border: 'none',
      cursor: 'pointer',
      backgroundColor: isDark ? '#ffffff' : '#000000',
      color: isDark ? '#000000' : '#ffffff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '24px',
    },
    chatPopup: {
      position: 'fixed' as const,
      bottom: '100px',
      right: '20px',
      width: 'min(90vw, 360px)',
      height: 'min(70vh, 480px)',
      backgroundColor: isDark ? '#000000' : '#ffffff',
      borderRadius: '12px',
      display: isChatOpen ? 'flex' : 'none',
      flexDirection: 'column' as const,
      zIndex: 1000,
      border: isDark ? '1px solid #333' : '1px solid #ddd',
    },
    chatHeader: {
      padding: '16px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      borderBottom: '1px solid #333',
      color: isDark ? '#ffffff' : '#000000',
    },
    messagesContainer: {
      flex: 1,
      padding: '16px',
      overflowY: 'auto' as const,
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '12px',
    },
    messageWrapper: (isBot: boolean) => ({
      display: 'flex',
      justifyContent: isBot ? 'flex-start' : 'flex-end',
    }),
    message: (isBot: boolean) => ({
      maxWidth: '70%',
      padding: '10px 14px',
      borderRadius: '12px',
      backgroundColor: isBot ? '#1a1a1a' : '#ffffff',
      color: isBot ? '#ffffff' : '#000000',
      fontSize: '14px',
    }),
    inputContainer: {
      padding: '16px',
      display: 'flex',
      gap: '8px',
    },
    input: {
      flex: 1,
      padding: '10px 14px',
      borderRadius: '8px',
      backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
      color: isDark ? '#ffffff' : '#000000',
      border: isDark ? '1px solid #333' : '1px solid #ddd',
      fontSize: '14px',
      outline: 'none',
    },
    sendButton: {
      padding: '10px 14px',
      borderRadius: '8px',
      cursor: loading ? 'not-allowed' : 'pointer',
      opacity: loading ? 0.6 : 1,
      backgroundColor: isDark ? '#1f2937' : '#e5e7eb',
      color: isDark ? '#ffffff' : '#000000',
      border: 'none',
      fontSize: '16px',
    },
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <button onClick={toggleTheme} style={styles.themeButton}>
          {isDark ? <Sun /> : <Moon />}
        </button>
      </div>

      <div style={styles.centerContent}>
        <h1 style={styles.heading}>TechGear Customer Support</h1>
      </div>

      <div style={styles.bottomSection}>
        <button onClick={handleChatbotClick} style={styles.chatButton}>
          <MessageCircle />
        </button>
      </div>

      <div style={styles.chatPopup}>
        <div style={styles.chatHeader}>
          <h3>Support Chat</h3>
          <button onClick={handleChatbotClick}><X /></button>
        </div>

        <div style={styles.messagesContainer}>
          {messages.map((m: Message, i: number) => (
            <div key={i} style={styles.messageWrapper(m.isBot)}>
              <div style={styles.message(m.isBot)}>{m.text}</div>
            </div>
          ))}
          {loading && (
            <div style={styles.messageWrapper(true)}>
              <div style={styles.message(true)}>Thinking…</div>
            </div>
          )}
        </div>

        <div style={styles.inputContainer}>
          <input
            value={inputValue}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message…"
            style={styles.input}
          />
          <button onClick={handleSendMessage} style={styles.sendButton}>
            <Send />
          </button>
        </div>
      </div>
    </div>
  );
}
