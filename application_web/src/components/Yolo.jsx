import { useState, useRef } from 'react'
import oracleData from './oracleData'

export default function Yolo({ baseUrl, db, apiBase }) {
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')
  const [hoveredBox, setHoveredBox] = useState(null)
  const [hoveredMeaning, setHoveredMeaning] = useState(null)
  const [showAIPanel, setShowAIPanel] = useState(false)
  const [aiCardVisible, setAiCardVisible] = useState({ aiInterpretation: false, aiComment: false })
  const [aiInterpretationText, setAiInterpretationText] = useState('')
  const [aiCommentText, setAiCommentText] = useState('')
  const [showInterpretationCursor, setShowInterpretationCursor] = useState(true)
  const [showCommentCursor, setShowCommentCursor] = useState(true)
  const imgRef = useRef(null)

  // 打字机效果函数
  const typeWriter = (text, setText, setShowCursor, baseDelay = 50) => {
    let index = 0
    const type = () => {
      if (index <= text.length) {
        setText(text.substring(0, index))
        index++
        if (index <= text.length) {
          const delay = baseDelay + Math.random() * 100
          setTimeout(type, delay)
        } else {
          setTimeout(() => setShowCursor(false), 1500)
        }
      }
    }
    type()
  }

  // 解析文本并渲染格式化的 React 元素
  const renderFormattedText = (text) => {
    const lines = text.split('\n')
    return lines.map((line, i) => {
      if (line.startsWith('#### ')) {
        return <h5 key={i} style={{ color: '#8E443D', margin: '16px 0 8px', fontWeight: 600 }}>{line.replace('#### ', '')}</h5>
      }
      if (line.startsWith('**') && line.endsWith('**')) {
        return <p key={i} style={{ margin: '4px 0' }}><strong style={{ color: '#511730' }}>{line.replace(/\*\*/g, '')}</strong></p>
      }
      if (line.startsWith('- ')) {
        return <p key={i} style={{ margin: '4px 0 4px 16px' }}>{line}</p>
      }
      if (line.match(/^\d+\./)) {
        return <p key={i} style={{ margin: '8px 0', fontWeight: 500, color: '#511730' }}>{line}</p>
      }
      return line ? <p key={i} style={{ margin: '4px 0' }}>{line}</p> : null
    })
  }

  // 解析 AI 点评格式
  const renderCommentText = (text) => {
    const lines = text.split('\n')
    return lines.map((line, i) => {
      if (line.startsWith('- **')) {
        const match = line.match(/^- \*\*(.+?)\*\*:?\s*(.*)$/)
        if (match) {
          return (
            <p key={i} style={{ margin: '8px 0 4px 16px' }}>
              <strong style={{ color: '#511730' }}>{match[1]}</strong>
              {match[2] && `: ${match[2]}`}
            </p>
          )
        }
      }
      if (line.startsWith('- ')) {
        return <p key={i} style={{ margin: '4px 0 4px 16px' }}>{line.replace('- ', '')}</p>
      }
      return line ? <p key={i} style={{ margin: '4px 0' }}>{line}</p> : null
    })
  }

  const moveCarousel = (dir) => {
    setIndex((index + dir + db.yolo.length) % db.yolo.length)
  }

  const runExample = async () => {
    setLoading(true)
    setStatus('正在通过 YOLOv8 检测甲骨文+ViT识别...')
    setResult(null)
    setHoveredBox(null)
    setHoveredMeaning(null)
    setShowAIPanel(false)
    setAiCardVisible({ aiInterpretation: false, aiComment: false })
    setAiInterpretationText('')
    setAiCommentText('')
    setShowInterpretationCursor(true)
    setShowCommentCursor(true)

    const formData = new FormData()
    formData.append('url', baseUrl + db.yolo[index])

    try {
      const res = await fetch(`${apiBase}/detect_yolo_with_vit`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setStatus(`处理完成 - 检测到 ${data.char_results?.length || 0} 个甲骨文`)
    } catch (e) {
      setStatus('错误: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFile = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setLoading(true)
    setStatus('正在处理本地图像...')
    setResult(null)
    setHoveredBox(null)
    setHoveredMeaning(null)
    setShowAIPanel(false)
    setAiCardVisible({ aiInterpretation: false, aiComment: false })
    setAiInterpretationText('')
    setAiCommentText('')
    setShowInterpretationCursor(true)
    setShowCommentCursor(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${apiBase}/detect_yolo_with_vit`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setStatus(`处理完成 - 检测到 ${data.char_results?.length || 0} 个甲骨文`)
    } catch (e) {
      setStatus('错误: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleMouseMove = (e, imgElement) => {
    if (!result || !result.detections || !imgElement) return

    const rect = imgElement.getBoundingClientRect()
    const scaleX = imgElement.naturalWidth / rect.width
    const scaleY = imgElement.naturalHeight / rect.height

    if (scaleX <= 0 || scaleY <= 0) return

    const mouseX = (e.clientX - rect.left) * scaleX
    const mouseY = (e.clientY - rect.top) * scaleY

    const charResults = result.char_results || []
    const detections = result.detections || []

    for (let i = 0; i < detections.length; i++) {
      const det = detections[i]
      const [x1, y1, x2, y2] = det.bbox

      if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
        const char = charResults[i]
        setHoveredBox(char && char.char_image ? char : null)
        if (currentData && currentData.charMeanings && currentData.charMeanings[i]) {
          // 使用检测框在数组中的索引匹配释义
          setHoveredMeaning(currentData.charMeanings[i])
        } else {
          setHoveredMeaning(null)
        }
        return
      }
    }
    setHoveredBox(null)
    setHoveredMeaning(null)
  }

  const currentYoloKey = `yolo${index + 1}`
  const currentData = oracleData[currentYoloKey]

  return (
    <div>
      <div style={{
        position: 'relative',
        width: '100%',
        overflow: 'hidden',
        borderRadius: '16px',
        background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.6))',
        border: '1px solid rgba(81, 23, 48, 0.1)',
        marginBottom: '28px',
        boxShadow: 'inset 0 0 30px rgba(81, 23, 48, 0.03)'
      }}>
        <button
          onClick={() => moveCarousel(-1)}
          style={{
            position: 'absolute',
            top: '50%',
            left: '16px',
            transform: 'translateY(-50%)',
            background: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid rgba(81, 23, 48, 0.15)',
            color: '#511730',
            width: '44px',
            height: '44px',
            fontSize: '20px',
            cursor: 'pointer',
            borderRadius: '50%',
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          ‹
        </button>

        <div style={{
          display: 'flex',
          transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
          transform: `translateX(-${index * 100}%)`
        }}>
          {db.yolo.map((img, i) => (
            <div key={i} style={{ minWidth: '100%', boxSizing: 'border-box', padding: '32px', textAlign: 'center' }}>
              <p style={{ fontWeight: 600, marginBottom: '16px', color: '#511730' }}>素材名: {img}</p>
              <img
                src={baseUrl + img}
                alt={img}
                style={{ maxHeight: '220px', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 2px 8px rgba(81, 23, 48, 0.08)', background: 'white', padding: '12px' }}
              />
            </div>
          ))}
        </div>

        <button
          onClick={() => moveCarousel(1)}
          style={{
            position: 'absolute',
            top: '50%',
            right: '16px',
            transform: 'translateY(-50%)',
            background: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid rgba(81, 23, 48, 0.15)',
            color: '#511730',
            width: '44px',
            height: '44px',
            fontSize: '20px',
            cursor: 'pointer',
            borderRadius: '50%',
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          ›
        </button>
      </div>

      <button
        onClick={runExample}
        disabled={loading}
        style={{
          background: 'linear-gradient(135deg, #8E443D 0%, #511730 100%)',
          color: 'white',
          border: 'none',
          padding: '16px 28px',
          borderRadius: '8px',
          cursor: loading ? 'not-allowed' : 'pointer',
          width: '100%',
          fontWeight: 600,
          fontSize: '15px',
          letterSpacing: '1px',
          boxShadow: '0 4px 16px rgba(81, 23, 48, 0.25)',
          marginBottom: '24px',
          opacity: loading ? 0.6 : 1
        }}
      >
        检测上方展示图片中的甲骨文
      </button>

      <div
        onClick={() => document.getElementById('file-yolo').click()}
        style={{
          border: '2px dashed rgba(81, 23, 48, 0.25)',
          padding: '32px',
          textAlign: 'center',
          marginBottom: '20px',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          borderRadius: '16px',
          background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(245,230,211,0.4))'
        }}
      >
        <span style={{ fontSize: '36px', marginBottom: '8px', display: 'block' }}>📂</span>
        或点击此处上传本地图片
        <input type="file" id="file-yolo" hidden accept="image/*" onChange={handleFile} />
      </div>

      <div style={{
        marginBottom: '20px',
        fontWeight: 600,
        minHeight: '24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px'
      }}>
        {loading && <div style={{
          width: '20px',
          height: '20px',
          border: '3px solid rgba(81, 23, 48, 0.1)',
          borderTopColor: '#8E443D',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite'
        }} />}
        {status}
      </div>

      {result && (
        <div style={{ marginTop: '28px' }}>
          <div style={{
            display: 'flex',
            gap: '20px',
            alignItems: 'stretch',
            flexWrap: 'wrap'
          }}>
            {/* 左侧：仅显示原拓片视图 */}
            <div style={{
              flex: '1 1 55%',
              minWidth: '280px',
              border: '1px solid rgba(81, 23, 48, 0.1)',
              borderRadius: '16px',
              background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5))',
              padding: '20px'
            }}>
              <h4 style={{ color: '#511730', fontSize: '15px', marginBottom: '12px', fontWeight: 600 }}>
                原拓片视图 - YOLOv8 检测结果
              </h4>
              <div style={{ textAlign: 'center' }}>
                <img
                  ref={imgRef}
                  src={'data:image/png;base64,' + result.original}
                  alt="原拓片"
                  style={{ maxWidth: '100%', maxHeight: '420px', borderRadius: '8px' }}
                  onMouseMove={(e) => handleMouseMove(e, e.target)}
                  onMouseLeave={() => { setHoveredBox(null); setHoveredMeaning(null) }}
                />
              </div>
            </div>

            {/* 右侧：悬停时显示释义 */}
            <div style={{
              flex: '0 0 220px',
              minWidth: '220px',
              border: '1px solid rgba(81, 23, 48, 0.1)',
              borderRadius: '16px',
              background: 'linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.5))',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'flex-start',
              minHeight: '280px'
            }}>
              <h4 style={{ color: '#511730', fontSize: '15px', marginBottom: '16px', fontWeight: 600 }}>
                悬停释义
              </h4>
              {hoveredMeaning ? (
                <div
                  style={{
                    width: '100%',
                    background: 'linear-gradient(180deg, #faf8f4, #f5f0e8)',
                    borderRadius: '12px',
                    padding: '20px 16px',
                    textAlign: 'center',
                    border: '1px solid rgba(81,23,48,0.1)'
                  }}
                >
                  <div style={{ marginBottom: '12px' }}>
                    <span
                      style={{
                        fontSize: '56px',
                        fontWeight: 700,
                        color: '#511730',
                        lineHeight: 1
                      }}
                    >
                      {hoveredMeaning.char}
                    </span>
                  </div>
                  {hoveredMeaning.pinyin && (
                    <div
                      style={{
                        fontSize: '18px',
                        color: '#CB9173',
                        marginBottom: '12px',
                        fontStyle: 'italic'
                      }}
                    >
                      {hoveredMeaning.pinyin}
                    </div>
                  )}
                  {hoveredMeaning.type && (
                    <div
                      style={{
                        display: 'inline-block',
                        fontSize: '12px',
                        color: '#8E443D',
                        background: 'rgba(81,23,48,0.08)',
                        padding: '4px 12px',
                        borderRadius: '12px',
                        marginBottom: '12px'
                      }}
                    >
                      {hoveredMeaning.type}
                    </div>
                  )}
                  <div
                    style={{
                      fontSize: '13px',
                      color: '#5A5A5A',
                      lineHeight: 1.6,
                      textAlign: 'left'
                    }}
                  >
                    {hoveredMeaning.desc}
                  </div>
                </div>
              ) : (
                <div style={{
                  width: '100%',
                  height: '200px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'rgba(81, 23, 48, 0.35)',
                  fontSize: '13px',
                  textAlign: 'center'
                }}>
                  移至检测框查看释义
                </div>
              )}
            </div>
          </div>

          {currentData && (
            <>
              <div
                style={{
                  marginTop: '28px',
                  border: '1px solid rgba(81, 23, 48, 0.12)',
                  borderRadius: '16px',
                  padding: '24px',
                  background: 'linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(245, 230, 211, 0.5))'
                }}
              >
                <h4
                  style={{
                    color: '#511730',
                    fontSize: '16px',
                    marginBottom: '16px',
                    fontWeight: 700,
                    textAlign: 'center',
                    letterSpacing: '1px'
                  }}
                >
                  文字释义
                </h4>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <thead>
                      <tr style={{ background: 'rgba(81, 23, 48, 0.06)' }}>
                        <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>检测框编号</th>
                        <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>拟定字名</th>
                        <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>字义说明</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentData.charMeanings.map((item, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid rgba(81, 23, 48, 0.06)' }}>
                          <td style={{ padding: '10px 8px', color: '#8E443D', fontWeight: 500 }}>{item.id}{item.position ? ` (${item.position})` : ''}</td>
                          <td style={{ padding: '10px 8px' }}>
                            <strong style={{ color: '#511730' }}>{item.char}</strong>
                            {item.pinyin && <span style={{ color: '#CB9173', marginLeft: '6px', fontSize: '12px' }}>{item.pinyin}</span>}
                            {item.type && <span style={{ color: '#8E443D', marginLeft: '8px', fontSize: '11px', background: 'rgba(81,23,48,0.06)', padding: '2px 6px', borderRadius: '4px' }}>{item.type}</span>}
                          </td>
                          <td style={{ padding: '10px 8px', color: '#5A5A5A', lineHeight: 1.5 }}>{item.desc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {!showAIPanel ? (
                <button
                  onClick={() => {
                    setShowAIPanel(true)
                    setAiCardVisible({ aiInterpretation: false, aiComment: false })
                    setAiInterpretationText('')
                    setAiCommentText('')
                    setShowInterpretationCursor(true)
                    setShowCommentCursor(true)
                    setTimeout(() => setAiCardVisible({ aiInterpretation: true, aiComment: false }), 100)
                    setTimeout(() => setAiCardVisible({ aiInterpretation: true, aiComment: true }), 800)
                    if (currentData) {
                      setTimeout(() => typeWriter(currentData.combinedMeaning, setAiInterpretationText, setShowInterpretationCursor, 30), 300)
                      setTimeout(() => typeWriter(currentData.aiComment, setAiCommentText, setShowCommentCursor, 40), 1800)
                    }
                  }}
                  style={{
                    marginTop: '20px',
                    width: '100%',
                    background: 'linear-gradient(135deg, #8E443D 0%, #511730 100%)',
                    color: 'white',
                    border: 'none',
                    padding: '14px 28px',
                    borderRadius: '10px',
                    cursor: 'pointer',
                    fontWeight: 700,
                    fontSize: '14px',
                    letterSpacing: '2px',
                    boxShadow: '0 4px 16px rgba(81, 23, 48, 0.25)',
                    transition: 'all 0.3s ease',
                    animation: 'fpPulse 2s ease-in-out infinite'
                  }}
                >
                  开始AI分析
                </button>
              ) : (
                <>
                  <div
                    style={{
                      marginTop: '24px',
                      border: '1px solid rgba(81, 23, 48, 0.12)',
                      borderRadius: '16px',
                      padding: '24px',
                      background: 'linear-gradient(145deg, rgba(224, 214, 138, 0.15), rgba(203, 145, 115, 0.1))',
                      opacity: aiCardVisible.aiInterpretation ? 1 : 0,
                      transform: aiCardVisible.aiInterpretation ? 'translateY(0) scale(1)' : 'translateY(30px) scale(0.95)',
                      transition: 'all 0.6s ease'
                    }}
                  >
                    <h4
                      style={{
                        color: '#511730',
                        fontSize: '16px',
                        marginBottom: '16px',
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        letterSpacing: '1px'
                      }}
                    >
                      <span style={{ background: 'linear-gradient(135deg, #8E443D, #511730)', color: '#fff', padding: '4px 12px', borderRadius: '6px', fontSize: '12px' }}>AI解读</span>
                      {currentData.title}
                    </h4>
                    <div style={{ color: '#5A5A5A', lineHeight: 1.8, fontSize: '14px', whiteSpace: 'pre-wrap' }}>
                      {renderFormattedText(aiInterpretationText)}
                      {showInterpretationCursor && (
                        <span style={{
                          display: 'inline-block',
                          width: '2px',
                          height: '16px',
                          background: '#511730',
                          marginLeft: '2px',
                          animation: 'cursorBlink 0.8s infinite'
                        }} />
                      )}
                    </div>
                  </div>

                  <div
                    style={{
                      marginTop: '24px',
                      border: '1px solid rgba(81, 23, 48, 0.12)',
                      borderRadius: '16px',
                      padding: '24px',
                      background: 'linear-gradient(145deg, rgba(203, 145, 115, 0.15), rgba(81, 23, 48, 0.08))',
                      opacity: aiCardVisible.aiComment ? 1 : 0,
                      transform: aiCardVisible.aiComment ? 'translateY(0) scale(1)' : 'translateY(30px) scale(0.95)',
                      transition: 'all 0.6s ease',
                      transitionDelay: '0.3s'
                    }}
                  >
                    <h4
                      style={{
                        color: '#511730',
                        fontSize: '16px',
                        marginBottom: '16px',
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        letterSpacing: '1px'
                      }}
                    >
                      <span style={{ background: 'linear-gradient(135deg, #CB9173, #8E443D)', color: '#fff', padding: '4px 12px', borderRadius: '6px', fontSize: '12px' }}>AI点评</span>
                      {currentData.title}
                    </h4>
                    <div style={{ color: '#5A5A5A', lineHeight: 1.8, fontSize: '14px', whiteSpace: 'pre-wrap' }}>
                      {renderCommentText(aiCommentText)}
                      {showCommentCursor && (
                        <span style={{
                          display: 'inline-block',
                          width: '2px',
                          height: '16px',
                          background: '#511730',
                          marginLeft: '2px',
                          animation: 'cursorBlink 0.8s infinite'
                        }} />
                      )}
                    </div>
                  </div>
                </>
              )}

              <style>{`
                @keyframes fpPulse {
                  0%, 100% { box-shadow: 0 4px 16px rgba(81, 23, 48, 0.25); }
                  50% { box-shadow: 0 4px 24px rgba(142, 68, 61, 0.4); }
                }
                @keyframes cursorBlink {
                  0%, 50% { opacity: 1; }
                  51%, 100% { opacity: 0; }
                }
                @keyframes textReveal {
                  from { opacity: 0; transform: translateX(-10px); }
                  to { opacity: 1; transform: translateX(0); }
                }
                .text-reveal {
                  opacity: 0;
                  animation: textReveal 0.4s ease forwards;
                }
              `}</style>
            </>
          )}
        </div>
      )}
    </div>
  )
}
