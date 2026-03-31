import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ChatBubbleLeftRightIcon, PaperAirplaneIcon, SparklesIcon } from '@heroicons/react/24/outline'
import { chatbotAPI } from '../services/api'
import VoiceMessagePlayer from './VoiceMessagePlayer'

const DEFAULT_SUGGESTED_QUESTIONS = [
  'What is this video about?',
  'Give me a short summary.',
  'What are the key takeaways?',
]

function formatTimestampLabel(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = seconds % 60

  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
  }

  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}

function parseSourceTimestamp(timestamp) {
  if (!timestamp) return null
  const raw = String(timestamp).trim()
  const startPart = raw.split(/\u2014|\u2013|-/)[0]?.trim()
  if (!startPart) return null

  if (startPart.includes(':')) {
    const parts = startPart.split(':').map((part) => Number(part.trim()))
    if (parts.some((part) => Number.isNaN(part))) return null
    if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2]
    if (parts.length === 2) return (parts[0] * 60) + parts[1]
    if (parts.length === 1) return parts[0]
  }

  const match = startPart.match(/(\d+(?:\.\d+)?)s?/)
  if (!match) return null
  const seconds = Number(match[1])
  return Number.isNaN(seconds) ? null : seconds
}

function normalizeSourceRange(timestamp) {
  if (!timestamp) return '00:00 \u2014 00:00'
  const raw = String(timestamp).trim()
  const parts = raw.split(/\u2014|\u2013|-/).map((part) => part.trim()).filter(Boolean)

  if (parts.length < 2) {
    const seconds = parseSourceTimestamp(raw)
    return `${formatTimestampLabel(seconds)} \u2014 ${formatTimestampLabel(seconds)}`
  }

  return `${formatTimestampLabel(parseSourceTimestamp(parts[0]))} \u2014 ${formatTimestampLabel(parseSourceTimestamp(parts[1]))}`
}

function parseAnswerContent(content) {
  const normalized = String(content || '').replace(/\r\n?/g, '\n').trim()
  if (!normalized) {
    return { explanation: '', keyPoints: [] }
  }

  const answerMatch = normalized.match(/Answer:\s*([\s\S]*?)(?:\n\s*Key Points:|$)/i)
  const keyMatch = normalized.match(/Key Points:\s*([\s\S]*)$/i)

  const explanation = (answerMatch ? answerMatch[1] : normalized)
    .replace(/\s+/g, ' ')
    .trim()

  const keyPoints = keyMatch
    ? keyMatch[1]
      .split('\n')
      .map((line) => line.replace(/^[\u2022*\-\s]+/, '').trim())
      .filter(Boolean)
    : []

  return { explanation, keyPoints }
}

function ChatBot({ videoId, momentContext = null, onClearMomentContext = null, onJumpToTimestamp = null }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [activeAudioUrl, setActiveAudioUrl] = useState(null)
  const [autoPlayAudioUrl, setAutoPlayAudioUrl] = useState(null)
  const [isRebuilding, setIsRebuilding] = useState(false)

  const [activeMomentContext, setActiveMomentContext] = useState(null)
  const messagesContainerRef = useRef(null)
  const previousMessageCountRef = useRef(0)
  const storageKey = videoId ? `chat_session_${videoId}` : null

  useEffect(() => {
    setMessages([])
    setSessionId(null)
    previousMessageCountRef.current = 0
    loadSuggestedQuestions()
    loadExistingConversation()
  }, [videoId])

  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    const shouldAnimate = previousMessageCountRef.current > 0 && messages.length > previousMessageCountRef.current
    if (shouldAnimate) {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
    } else {
      container.scrollTop = container.scrollHeight
    }

    previousMessageCountRef.current = messages.length
  }, [messages])

  useEffect(() => {
    if (!momentContext) return
    setActiveMomentContext(momentContext)
    setInput((prev) => (prev.trim() ? prev : 'What is happening at this moment?'))
  }, [momentContext])

  const loadSuggestedQuestions = async () => {
    try {
      const response = await chatbotAPI.getSuggestedQuestions(videoId)
      const questions = Array.isArray(response?.data?.questions) ? response.data.questions : []
      setSuggestedQuestions(questions.length > 0 ? questions : DEFAULT_SUGGESTED_QUESTIONS)
    } catch (error) {
      console.error('Failed to load suggested questions:', error)
      setSuggestedQuestions(DEFAULT_SUGGESTED_QUESTIONS)
    }
  }

  const handleRebuildIndex = async () => {
    if (!videoId) return
    setIsRebuilding(true)
    try {
      await chatbotAPI.rebuildIndex(videoId)
      alert('Index rebuilt successfully! The chatbot should now work better.')
      loadSuggestedQuestions()
    } catch (error) {
      console.error('Failed to rebuild index:', error)
      alert('Failed to rebuild index. Please try again.')
    } finally {
      setIsRebuilding(false)
    }
  }

  const normalizeSessionList = (data) => {
    if (Array.isArray(data)) return data
    if (data && Array.isArray(data.results)) return data.results
    return []
  }

  const loadExistingConversation = async () => {
    if (!videoId) return

    try {
      const persisted = storageKey ? localStorage.getItem(storageKey) : null
      let activeSessionId = persisted

      if (!activeSessionId) {
        const sessionsResp = await chatbotAPI.getSessions({ video_id: videoId })
        const sessions = normalizeSessionList(sessionsResp.data)
        if (sessions.length > 0) {
          activeSessionId = sessions[0].id
        }
      }

      if (!activeSessionId) return

      const messagesResp = await chatbotAPI.getSessionMessages(activeSessionId)
      const history = Array.isArray(messagesResp.data) ? messagesResp.data : []

      const mapped = history.map((message) => ({
        role: message.sender === 'user' ? 'user' : 'bot',
        content: message.message,
        sources: message.referenced_segments || [],
        audioUrl: message.audio_url || null,
      }))

      setSessionId(activeSessionId)
      setMessages(mapped)
      if (storageKey) localStorage.setItem(storageKey, activeSessionId)
    } catch (error) {
      console.error('Failed to load existing conversation:', error)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setLoading(true)
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])

    try {
      const payload = {
        video_id: videoId,
        message: userMessage,
        session_id: sessionId,
        strict_mode: true,
        generate_tts: true,
        context_timestamp: activeMomentContext?.timestamp ?? null,
      }

      if (typeof activeMomentContext?.windowSeconds === 'number') {
        payload.context_window_seconds = activeMomentContext.windowSeconds
      }

      const response = await chatbotAPI.sendMessage(payload)

      if (response.data.session_id) {
        setSessionId(response.data.session_id)
        if (storageKey) {
          localStorage.setItem(storageKey, response.data.session_id)
        }
      }

      // Use regular answer
      const answerContent = response.data.answer
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: answerContent,
          sources: response.data.sources || [],
          audioUrl: response.data.audio_url || null,
          timestampContext: response.data.timestamp_context || null,
        },
      ])
      setAutoPlayAudioUrl(response.data.audio_url || null)
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: 'Sorry, I failed to process your question. Please try again.',
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  const handleSuggestedQuestion = (question) => {
    setInput(question)
  }

  const handleSourceClick = (source) => {
    const seconds = parseSourceTimestamp(source?.timestamp)
    if (seconds == null || !onJumpToTimestamp) return
    onJumpToTimestamp(seconds)
  }

  return (
    <motion.div
      className="chatbot-shell flex flex-col h-[500px]"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div className="chatbot-shell-header flex items-center justify-between p-4">
        <div className="flex items-center space-x-3">
          <motion.div
            className="p-2 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl"
            whileHover={{ scale: 1.1 }}
          >
            <ChatBubbleLeftRightIcon className="w-6 h-6 text-indigo-400" />
          </motion.div>
          <div>
            <h3 className="text-lg font-semibold text-white">Video Chatbot</h3>
            <p className="text-xs text-white/50">
              {activeMomentContext ? `Focused on ${activeMomentContext.label}` : 'Ask questions about this video'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRebuildIndex}
            disabled={isRebuilding}
            className="chatbot-shell-action px-3 py-1.5 text-xs rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-50"
            title="Rebuild the search index if chatbot isn't working properly"
          >
            <svg className={`w-3.5 h-3.5 ${isRebuilding ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {isRebuilding ? 'Rebuilding...' : 'Rebuild Index'}
          </button>
        </div>
      </div>

      {activeMomentContext && (
        <div className="mx-4 mt-4 p-3 rounded-xl border border-fuchsia-400/20 bg-fuchsia-500/10">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-mono text-fuchsia-300 mb-1">{activeMomentContext.label}</p>
              <p className="text-sm text-white/80 max-h-10 overflow-hidden">{activeMomentContext.excerpt}</p>
            </div>
            <button
              type="button"
              onClick={() => {
                setActiveMomentContext(null)
                if (onClearMomentContext) onClearMomentContext()
              }}
              className="text-xs text-white/50 hover:text-white"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      <div ref={messagesContainerRef} className="chatbot-shell-messages flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {messages.length === 0 && (
          <motion.div
            className="flex flex-col items-center justify-center h-full text-center py-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mb-4"
              animate={{
                y: [0, -10, 0],
                scale: [1, 1.05, 1],
              }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <SparklesIcon className="w-8 h-8 text-indigo-400" />
            </motion.div>
            <p className="text-white/60 mb-2">Ask me anything about this video!</p>
            <p className="text-xs text-white/40">I can summarize, answer questions, or explain topics.</p>
          </motion.div>
        )}

        {messages.map((message, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={message.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
          >
            <div className={message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot'}>
              <div className="flex justify-between items-start gap-2">
                {message.role === 'bot' ? (
                  <div className="flex-1 min-w-0">
                    {(() => {
                      const parsed = parseAnswerContent(message.content)
                      return (
                        <div className="chat-answer-layout">
                          <div className="chat-answer-section-header">Answer</div>
                          <div className="chat-answer-divider" />
                          <p className="chat-answer-text">{parsed.explanation || message.content}</p>

                          {parsed.keyPoints.length > 0 && (
                            <div className="chat-keypoints-block">
                              <div className="chat-keypoints-title">Key Points</div>
                              <ul className="chat-keypoints-list">
                                {parsed.keyPoints.map((point, pointIndex) => (
                                  <li key={`${index}-point-${pointIndex}`} className="chat-keypoint-item">
                                    <span className="chat-keypoint-bullet">{'\u2022'}</span>
                                    <span>{point}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {message.audioUrl && (
                            <VoiceMessagePlayer
                              audioUrl={message.audioUrl}
                              activeAudioUrl={activeAudioUrl}
                              onSetActiveAudioUrl={setActiveAudioUrl}
                              autoPlay={message.audioUrl === autoPlayAudioUrl}
                            />
                          )}
                        </div>
                      )
                    })()}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap flex-1">{message.content}</p>
                )}
              </div>

              {message.sources && message.sources.length > 0 && (
                <div className="source-reference">
                  {message.timestampContext && (
                    <p className="text-[11px] text-fuchsia-300 mb-2">Moment Context: {message.timestampContext}</p>
                  )}
                  <p className="chat-sources-title">Sources from this video</p>
                  {message.sources.map((source, sourceIndex) => (
                    <motion.button
                      type="button"
                      key={sourceIndex}
                      className={`source-item source-item-card mb-2 w-full text-left ${onJumpToTimestamp ? 'cursor-pointer' : ''}`}
                      whileHover={{ x: 5 }}
                      onClick={() => handleSourceClick(source)}
                      disabled={!onJumpToTimestamp}
                    >
                      <span className="source-item-time">{normalizeSourceRange(source.timestamp)}</span>
                      <span className="source-item-preview">
                        {source.text?.trim() || 'Transcript preview unavailable.'}
                      </span>
                    </motion.button>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ))}

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="chat-bubble-bot">
              <div className="typing-indicator">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {messages.length === 0 && suggestedQuestions.length > 0 && (
        <div className="chatbot-shell-footer p-4">
          <p className="text-xs text-white/40 mb-3">Suggested questions:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.slice(0, 3).map((question, index) => (
              <motion.button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                className="suggestion-chip"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                {question}
              </motion.button>
            ))}
          </div>
        </div>
      )}

      <div className="chatbot-shell-footer p-4">
        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your question..."
            className="chatbot-shell-input flex-1"
            disabled={loading}
          />
          <motion.button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="p-3 glow-button flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            whileHover={!input.trim() || loading ? {} : { scale: 1.05 }}
            whileTap={!input.trim() || loading ? {} : { scale: 0.95 }}
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  )
}

export default ChatBot

