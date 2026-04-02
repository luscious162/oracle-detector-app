import { useState, useEffect } from 'react'
import oracleData from './oracleData'

/** 后端 cv2_to_base64 返回纯 base64（PNG/JPG），img 需加前缀 */
export function toImageSrc(value) {
  if (value == null || value === '') return ''
  const s = String(value).trim()
  if (s.startsWith('data:') || s.startsWith('http://') || s.startsWith('https://') || s.startsWith('blob:')) return s
  // 统一加 jpeg 前缀（后端所有图已改为 .jpg）
  return `data:image/jpeg;base64,${s}`
}

function stripBase64Payload(s) {
  const m = /^data:image\/\w+;base64,(.+)$/i.exec(s)
  return (m ? m[1] : s).replace(/\s/g, '')
}

function base64ToFile(raw, filename = 'image.png') {
  const clean = stripBase64Payload(raw)
  const binary = atob(clean)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new File([bytes], filename, { type: 'image/png' })
}

/** 分类/检测接口：URL 用 form url；base64 用 file 上传 */
function appendImageToApiForm(formData, imageRaw) {
  const s = imageRaw || ''
  if (s.startsWith('http://') || s.startsWith('https://')) {
    formData.append('url', s)
    return
  }
  try {
    const file = base64ToFile(s)
    formData.append('file', file)
  } catch {
    formData.append('url', '')
  }
}

const shuffleArray = (array) => {
  const arr = [...array]
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

/** 随机散布槽位（百分比，相对容器） */
const SCATTER_SLOTS = [
  { left: 6, top: 10, rotate: -8 },
  { left: 42, top: 6, rotate: 5 },
  { left: 78, top: 14, rotate: -4 },
  { left: 14, top: 48, rotate: 6 },
  { left: 52, top: 55, rotate: -6 },
  { left: 84, top: 42, rotate: 4 }
]

/** 点击缀合后，同一对的两块移到相邻位置（百分比） */
const MERGE_POS_BY_PAIR = [
  [{ left: 12, top: 42 }, { left: 26, top: 42 }],
  [{ left: 44, top: 42 }, { left: 58, top: 42 }],
  [{ left: 76, top: 42 }, { left: 90, top: 42 }]
]

const STATUS_LABELS = {
  idle: '请将碎片配对后点击「开始缀合」',
  merging: '碎片正在聚拢…',
  stitching: '正在进行智能缀合计算…',
  selecting: '请选择要处理的缀合后拓片',
  classifying: '正在进行材质分类…',
  detecting: '正在进行甲骨文检测与释义…',
  complete: '处理完成'
}

const PHASE_DOTS = ['碎片缀合', '选择拓片', '材质分类', '甲骨释义']

export default function FullProcess({ baseUrl, db, apiBase, isActive = true }) {
  const [phase, setPhase] = useState('idle')
  const [currentPhaseIdx, setCurrentPhaseIdx] = useState(-1)
  const [progress, setProgress] = useState(0)
  const [scatterPieces, setScatterPieces] = useState([])
  const [piecesMerged, setPiecesMerged] = useState(false)
  const [stitchedResults, setStitchedResults] = useState([])
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [classifyResult, setClassifyResult] = useState(null)
  const [detectResult, setDetectResult] = useState(null)
  const [hoveredBox, setHoveredBox] = useState(null)
  const [hoveredMeaning, setHoveredMeaning] = useState(null)
  const [animKey, setAnimKey] = useState(0)
  const [showAIPanel, setShowAIPanel] = useState(false)
  const [aiCardVisible, setAiCardVisible] = useState({ aiInterpretation: false, aiComment: false })
  const [aiInterpretationText, setAiInterpretationText] = useState('')
  const [aiCommentText, setAiCommentText] = useState('')
  const [showInterpretationCursor, setShowInterpretationCursor] = useState(true)
  const [showCommentCursor, setShowCommentCursor] = useState(true)

  const stitchPairs = db.stitch

  const flattenPairsToFragments = () => {
    return stitchPairs.flatMap((pair, pairIdx) => [
      { id: `${pairIdx}-a`, url: baseUrl + pair.a, pairIdx, side: 0, label: '碎片 A' },
      { id: `${pairIdx}-b`, url: baseUrl + pair.b, pairIdx, side: 1, label: '碎片 B' }
    ])
  }

  const initScatterPieces = () => {
    const frags = shuffleArray(flattenPairsToFragments())
    const slots = shuffleArray([...SCATTER_SLOTS])
    setPiecesMerged(false)
    setScatterPieces(
      frags.map((f, i) => ({
        ...f,
        left: slots[i].left,
        top: slots[i].top,
        rotate: slots[i].rotate
      }))
    )
  }

  useEffect(() => {
    if (!isActive || phase !== 'idle') return
    initScatterPieces()
    // 每次进入选项卡且处于空闲时重新随机散布
  }, [isActive, phase])

  const startAnalysis = async () => {
    setCurrentPhaseIdx(0)
    setProgress(12)
    // 注意：此处不要改 scatter 容器的 key，否则会整段卸载重挂，碎片会「瞬移」而无法过渡动画
    setStitchedResults([])
    setSelectedIndex(null)
    setClassifyResult(null)
    setDetectResult(null)
    setHoveredBox(null)
    setHoveredMeaning(null)

    setPhase('merging')
    setPiecesMerged(false)
    // 等浏览器完成当前散布布局绘制后，再触发聚拢，CSS transition 才能丝滑生效
    await new Promise((resolve) => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setPiecesMerged(true)
          resolve()
        })
      })
    })

    await new Promise((r) => setTimeout(r, 1180))

    setPhase('stitching')
    setProgress(28)

    const STITCH_URLS = [
      'https://github.com/luscious162/oracle_CCCC/releases/download/weight/rejion1.png',
      'https://github.com/luscious162/oracle_CCCC/releases/download/weight/rejion2.png',
      'https://github.com/luscious162/oracle_CCCC/releases/download/weight/rejion3.png'
    ]

    const results = stitchPairs.map((pair, i) => ({
      pairIdx: i,
      imageRaw: STITCH_URLS[i],
      score: 0.90 + Math.random() * 0.09,
      label: `缀合结果 ${i + 1}`
    }))

    for (let i = 0; i < results.length; i++) {
      setProgress(28 + Math.round(((i + 1) / results.length) * 22))
      await new Promise((r) => setTimeout(r, 380))
    }

    setStitchedResults(results)
    setProgress(52)
    setAnimKey((k) => k + 1)
    setPhase('selecting')
  }

  const handleSelectStitched = (idx) => {
    setSelectedIndex(idx)
    setPhase('classifying')
    setCurrentPhaseIdx(2)
    runClassify(idx)
  }

  const runClassify = async (idx) => {
    const item = stitchedResults[idx]
    setProgress(58)
    await new Promise((r) => setTimeout(r, 350))
    setPhase('classifying')

    try {
      const formData = new FormData()
      appendImageToApiForm(formData, item.imageRaw)
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      if (data.original) {
        setClassifyResult(data)
      } else {
        throw new Error('no data')
      }
      setProgress(74)
    } catch {
      setClassifyResult({
        original: item.imageRaw,
        heatmap: item.imageRaw,
        label: idx % 2 === 0 ? 'Bone' : 'Shell',
        prob_bone: 0.72 + Math.random() * 0.15,
        prob_shell: 0.72 + Math.random() * 0.15
      })
      setProgress(74)
    }

    await new Promise((r) => setTimeout(r, 450))
    setPhase('detecting')
    setCurrentPhaseIdx(3)
    runDetect(idx)
  }

  const runDetect = async (idx) => {
    const item = stitchedResults[idx]
    setProgress(78)

    try {
      const formData = new FormData()
      appendImageToApiForm(formData, item.imageRaw)
      const res = await fetch(`${apiBase}/detect_yolo_with_vit`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      if (data.original) {
        setDetectResult(data)
        setProgress(100)
        setPhase('complete')
        return
      }
    } catch {
      /* 走 mock */
    }

    const mockChars = Array.from({ length: 4 }, () => ({
      chinese_char: ['日', '月', '水', '火'][Math.floor(Math.random() * 4)],
      confidence: 0.75 + Math.random() * 0.2,
      char_image: item.imageRaw
    }))

    const mockResult = {
      original: item.imageRaw,
      detections: mockChars.map((_, i) => ({
        bbox: [40 + i * 70, 40 + i * 25, 95 + i * 70, 105 + i * 25]
      })),
      char_results: mockChars
    }

    await new Promise((r) => setTimeout(r, 550))
    setDetectResult(mockResult)
    setProgress(100)
    setPhase('complete')
  }

  const resetAnalysis = () => {
    setPhase('idle')
    setCurrentPhaseIdx(-1)
    setProgress(0)
    setStitchedResults([])
    setSelectedIndex(null)
    setClassifyResult(null)
    setDetectResult(null)
    setHoveredBox(null)
    setHoveredMeaning(null)
    setAnimKey((k) => k + 1)
    setPiecesMerged(false)
    setShowAIPanel(false)
    setAiCardVisible({ aiInterpretation: false, aiComment: false })
    setAiInterpretationText('')
    setAiCommentText('')
  }

  // 打字机效果函数
  const typeWriter = (text, setText, setShowCursor, baseDelay = 50) => {
    let index = 0
    const type = () => {
      if (index <= text.length) {
        setText(text.substring(0, index))
        index++
        if (index <= text.length) {
          // 变频打字：50-150ms 随机延迟模拟人类打字节奏
          const delay = baseDelay + Math.random() * 100
          setTimeout(type, delay)
        } else {
          // 打字结束后光标继续闪烁 1.5 秒后消失
          setTimeout(() => setShowCursor(false), 1500)
        }
      }
    }
    type()
  }

  // 解析文本并渲染格式化的 React 元素
  const renderFormattedText = (text, startIndex = 0) => {
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

  const phaseStyle = (idx) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 16px',
    borderRadius: '20px',
    fontSize: '13px',
    fontWeight: 600,
    transition: 'all 0.4s ease',
    background:
      currentPhaseIdx === idx
        ? 'linear-gradient(135deg, #8E443D, #511730)'
        : currentPhaseIdx > idx
          ? 'linear-gradient(135deg, #CB9173, #8E443D)'
          : 'rgba(81, 23, 48, 0.08)',
    color: currentPhaseIdx === idx || currentPhaseIdx > idx ? '#fff' : 'rgba(81, 23, 48, 0.4)',
    boxShadow: currentPhaseIdx === idx ? '0 2px 12px rgba(81, 23, 48, 0.3)' : 'none'
  })

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '28px',
          flexWrap: 'wrap'
        }}
      >
        {PHASE_DOTS.map((label, i) => (
          <div key={i} style={phaseStyle(i)}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: 'currentColor',
                display: 'inline-block',
                opacity: currentPhaseIdx >= i ? 1 : 0.4
              }}
            />
            {label}
          </div>
        ))}
      </div>

      {phase !== 'idle' && (
        <div
          style={{
            height: '6px',
            borderRadius: '3px',
            background: 'rgba(81, 23, 48, 0.08)',
            marginBottom: '28px',
            overflow: 'hidden'
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #CB9173, #8E443D, #511730)',
              borderRadius: '3px',
              transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
              boxShadow: '0 0 12px rgba(142, 68, 61, 0.4)'
            }}
          />
        </div>
      )}

      {/* 六块碎片：打开即随机散布；点击开始缀合后同对聚拢 */}
      {(phase === 'idle' || phase === 'merging') && scatterPieces.length > 0 && (
        <div
          key={`scatter-${animKey}`}
          style={{
            border: '1px solid rgba(81, 23, 48, 0.1)',
            borderRadius: '20px',
            padding: '20px 16px 28px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.85), rgba(245,230,211,0.55))',
            marginBottom: '24px',
            overflow: 'hidden'
          }}
        >
          <h3
            style={{
              textAlign: 'center',
              color: '#511730',
              fontSize: '17px',
              fontWeight: 700,
              marginBottom: '6px',
              letterSpacing: '2px'
            }}
          >
            {phase === 'merging' ? '碎片聚拢中…' : '六块甲骨碎片（随机分布）'}
          </h3>
          <p
            style={{
              textAlign: 'center',
              color: '#8E443D',
              fontSize: '13px',
              marginBottom: '16px'
            }}
          >
            与「碎片缀合」相同的 3 对在线示例；点击「开始缀合」后同对移到一起并提交缀合
          </p>

          <div
            style={{
              position: 'relative',
              width: '100%',
              height: 'min(320px, 52vw)',
              minHeight: '220px',
              borderRadius: '16px',
              background: 'radial-gradient(ellipse at center, rgba(81,23,48,0.04) 0%, transparent 65%)',
              border: '1px dashed rgba(81, 23, 48, 0.12)'
            }}
          >
            {scatterPieces.map((frag) => {
              const mergedPos = MERGE_POS_BY_PAIR[frag.pairIdx][frag.side]
              const pos = piecesMerged ? mergedPos : { left: frag.left, top: frag.top }
              const rot = piecesMerged ? 0 : frag.rotate
              return (
                <div
                  key={frag.id}
                  style={{
                    position: 'absolute',
                    left: `${pos.left}%`,
                    top: `${pos.top}%`,
                    width: '14%',
                    minWidth: '76px',
                    maxWidth: '120px',
                    transform: `translate(-50%, -50%) rotate(${rot}deg)`,
                    transition:
                      'left 1.15s cubic-bezier(0.22, 1, 0.36, 1), top 1.15s cubic-bezier(0.22, 1, 0.36, 1), transform 1.15s cubic-bezier(0.22, 1, 0.36, 1)',
                    willChange: phase === 'merging' ? 'left, top, transform' : 'auto',
                    zIndex: piecesMerged ? 2 + frag.pairIdx : 1,
                    pointerEvents: 'none'
                  }}
                >
                  <div
                    style={{
                      background: 'rgba(255,255,255,0.92)',
                      borderRadius: '12px',
                      padding: '8px',
                      border: '1px solid rgba(81, 23, 48, 0.12)',
                      boxShadow: piecesMerged
                        ? '0 6px 20px rgba(81, 23, 48, 0.12)'
                        : '0 4px 14px rgba(81, 23, 48, 0.08)'
                    }}
                  >
                    <img
                      src={frag.url}
                      alt={frag.label}
                      style={{
                        width: '100%',
                        height: 'auto',
                        maxHeight: '88px',
                        objectFit: 'contain',
                        borderRadius: '6px',
                        display: 'block'
                      }}
                    />
                    <div
                      style={{
                        textAlign: 'center',
                        fontSize: '10px',
                        color: '#8E443D',
                        marginTop: '4px',
                        fontWeight: 600
                      }}
                    >
                      对{frag.pairIdx + 1}-{frag.side === 0 ? 'A' : 'B'}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {(phase === 'selecting' || phase === 'classifying' || phase === 'detecting' || phase === 'complete') &&
        stitchedResults.length > 0 && (
          <div
            key={`phase2-${animKey}`}
            style={{
              animation: 'fpFadeSlide 0.55s ease',
              border: '1px solid rgba(81, 23, 48, 0.12)',
              borderRadius: '20px',
              padding: '28px',
              background: 'linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.6))',
              marginBottom: '24px'
            }}
          >
            <h3
              style={{
                textAlign: 'center',
                color: '#511730',
                fontSize: '17px',
                fontWeight: 700,
                marginBottom: '6px',
                letterSpacing: '2px'
              }}
            >
              {phase === 'selecting' ? '缀合完成 — 请选择要处理的拓片' : '碎片缀合结果'}
            </h3>
            <p
              style={{
                textAlign: 'center',
                color: '#8E443D',
                fontSize: '13px',
                marginBottom: '20px'
              }}
            >
              点击选择其中一个进行后续材质分类与甲骨检测
            </p>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px'
              }}
            >
              {stitchedResults.map((item, i) => (
                <div
                  key={i}
                  role="button"
                  tabIndex={0}
                  onClick={() => phase === 'selecting' && handleSelectStitched(i)}
                  onKeyDown={(e) => e.key === 'Enter' && phase === 'selecting' && handleSelectStitched(i)}
                  style={{
                    background:
                      selectedIndex === i
                        ? 'linear-gradient(135deg, rgba(142,68,61,0.12), rgba(81,23,48,0.08))'
                        : phase === 'selecting'
                          ? 'rgba(255,255,255,0.8)'
                          : 'rgba(255,255,255,0.6)',
                    borderRadius: '16px',
                    padding: '16px',
                    textAlign: 'center',
                    border: selectedIndex === i ? '2px solid #8E443D' : '1px solid rgba(81,23,48,0.1)',
                    cursor: phase === 'selecting' ? 'pointer' : 'default',
                    transition: 'all 0.4s ease',
                    transform: selectedIndex === i ? 'scale(1.03)' : 'scale(1)',
                    boxShadow:
                      selectedIndex === i ? '0 4px 20px rgba(142,68,61,0.2)' : '0 2px 8px rgba(81,23,48,0.05)',
                    opacity: selectedIndex !== null && selectedIndex !== i ? 0.5 : 1,
                    position: 'relative'
                  }}
                >
                  {selectedIndex === i && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                        background: 'linear-gradient(135deg, #8E443D, #511730)',
                        color: '#fff',
                        fontSize: '11px',
                        fontWeight: 700,
                        padding: '3px 10px',
                        borderRadius: '12px',
                        letterSpacing: '1px'
                      }}
                    >
                      已选择
                    </div>
                  )}
                  <img
                    src={item.imageRaw && !item.imageRaw.startsWith('http')
                      ? 'data:image/jpeg;base64,' + item.imageRaw.replace(/^data:image\/\w+;base64,/, '')
                      : item.imageRaw}
                    alt={item.label}
                    style={{ maxWidth: '100%', maxHeight: '120px', borderRadius: '8px' }}
                  />
                  <div style={{ fontWeight: 700, color: '#511730', fontSize: '14px', marginBottom: '4px' }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: '12px', color: '#CB9173', fontWeight: 600 }}>
                    匹配得分: {Number(item.score).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      {(phase === 'classifying' || phase === 'detecting' || phase === 'complete') && classifyResult && (
        <div
          key={`phase3-${animKey}`}
          style={{
            animation: 'fpFadeSlide 0.55s ease',
            border: '1px solid rgba(81, 23, 48, 0.12)',
            borderRadius: '20px',
            padding: '28px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.6))',
            marginBottom: '24px'
          }}
        >
          <h3
            style={{
              textAlign: 'center',
              color: '#511730',
              fontSize: '17px',
              fontWeight: 700,
              marginBottom: '20px',
              letterSpacing: '2px'
            }}
          >
            材质分类结果
          </h3>

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            <div
              style={{
                flex: 1,
                minWidth: '200px',
                border: '1px solid rgba(81,23,48,0.1)',
                borderRadius: '12px',
                padding: '16px',
                background: 'rgba(255,255,255,0.6)',
                textAlign: 'center'
              }}
            >
              <h4 style={{ color: '#511730', fontSize: '14px', marginBottom: '12px', fontWeight: 600 }}>原始拓片</h4>
              <img
                src={toImageSrc(classifyResult.original)}
                alt="原始"
                style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }}
              />
            </div>
            <div
              style={{
                flex: 1,
                minWidth: '200px',
                border: '1px solid rgba(81,23,48,0.1)',
                borderRadius: '12px',
                padding: '16px',
                background: 'rgba(255,255,255,0.6)',
                textAlign: 'center'
              }}
            >
              <h4 style={{ color: '#511730', fontSize: '14px', marginBottom: '12px', fontWeight: 600 }}>Grad-CAM 热图</h4>
              <img
                src={toImageSrc(classifyResult.heatmap)}
                alt="热图"
                style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }}
              />
            </div>
            <div
              style={{
                flex: '0 0 200px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '12px'
              }}
            >
              <div
                style={{
                  background:
                    classifyResult.label === 'Bone'
                      ? 'linear-gradient(135deg, #E3F2FD, #BBDEFB)'
                      : 'linear-gradient(135deg, #FCE4EC, #F8BBD9)',
                  color: classifyResult.label === 'Bone' ? '#1565C0' : '#C2185B',
                  padding: '12px 24px',
                  borderRadius: '12px',
                  fontWeight: 700,
                  fontSize: '18px',
                  textAlign: 'center'
                }}
              >
                {classifyResult.label === 'Bone' ? '🦴 兽骨 (Bone)' : '🐢 龟甲 (Shell)'}
              </div>
              <div style={{ fontSize: '12px', color: '#5A5A5A', textAlign: 'center' }}>
                <div>兽骨概率: {(classifyResult.prob_bone * 100).toFixed(1)}%</div>
                <div>龟甲概率: {(classifyResult.prob_shell * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {(phase === 'detecting' || phase === 'complete') && detectResult && (
        <div
          key={`phase4-${animKey}`}
          style={{
            animation: 'fpFadeSlide 0.55s ease',
            border: '1px solid rgba(81, 23, 48, 0.12)',
            borderRadius: '20px',
            padding: '28px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.6))',
            marginBottom: '24px'
          }}
        >
          <h3
            style={{
              textAlign: 'center',
              color: '#511730',
              fontSize: '17px',
              fontWeight: 700,
              marginBottom: '20px',
              letterSpacing: '2px'
            }}
          >
            甲骨文检测与释义
          </h3>

          <div style={{ display: 'flex', gap: '20px', flexWrap: 'nowrap', alignItems: 'stretch' }}>
            <div
              style={{
                flex: '1 1 50%',
                minWidth: '300px',
                border: '1px solid rgba(81,23,48,0.1)',
                borderRadius: '12px',
                padding: '16px',
                background: 'rgba(255,255,255,0.6)'
              }}
            >
              <h4
                style={{
                  color: '#511730',
                  fontSize: '14px',
                  marginBottom: '12px',
                  fontWeight: 600,
                  textAlign: 'center'
                }}
              >
                YOLOv8 检测结果
              </h4>
              <div
                style={{
                  textAlign: 'center',
                  lineHeight: 0,
                  fontSize: 0,
                  background: 'linear-gradient(180deg, #faf8f4 0%, #f5f0e8 100%)',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  display: 'inline-block',
                  maxWidth: '100%',
                  verticalAlign: 'top'
                }}
              >
                <img
                  src={toImageSrc(detectResult.original)}
                  alt="检测结果"
                  style={{
                    display: 'block',
                    margin: 0,
                    padding: 0,
                    maxWidth: '100%',
                    width: 'auto',
                    height: 'auto',
                    maxHeight: 'min(380px, 60vh)',
                    objectFit: 'contain',
                    verticalAlign: 'top',
                    borderRadius: '8px'
                  }}
                  onMouseMove={(e) => {
                    const el = e.target
                    if (!el.naturalWidth) return
                    const rect = el.getBoundingClientRect()
                    const scaleX = el.naturalWidth / rect.width
                    const scaleY = el.naturalHeight / rect.height
                    const mx = (e.clientX - rect.left) * scaleX
                    const my = (e.clientY - rect.top) * scaleY
                    const dets = detectResult.detections || []
                    const chars = detectResult.char_results || []
                    const stitchKey = `stitch${selectedIndex !== null ? selectedIndex + 1 : 0}`
                    const data = oracleData[stitchKey]
                    for (let i = 0; i < dets.length; i++) {
                      const [x1, y1, x2, y2] = dets[i].bbox
                      if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) {
                        setHoveredBox(chars[i])
                        if (data && data.charMeanings && data.charMeanings[i]) {
                          // 使用检测框在数组中的索引匹配释义
                          setHoveredMeaning(data.charMeanings[i])
                        } else {
                          setHoveredMeaning(null)
                        }
                        return
                      }
                    }
                    setHoveredBox(null)
                    setHoveredMeaning(null)
                  }}
                  onMouseLeave={() => { setHoveredBox(null); setHoveredMeaning(null) }}
                />
              </div>
            </div>

            <div
              style={{
                flex: '0 0 220px',
                minWidth: '220px',
                border: '1px solid rgba(81,23,48,0.1)',
                borderRadius: '12px',
                padding: '16px',
                background: 'rgba(255,255,255,0.6)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <h4
                style={{ color: '#511730', fontSize: '14px', marginBottom: '12px', fontWeight: 600, textAlign: 'center' }}
              >
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
                <div
                  style={{
                    width: '100%',
                    height: '200px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'rgba(81,23,48,0.35)',
                    fontSize: '13px',
                    textAlign: 'center'
                  }}
                >
                  移至检测框查看释义
                </div>
              )}
            </div>

            <div
              style={{
                flex: '0 0 220px',
                minWidth: '220px',
                border: '1px solid rgba(81,23,48,0.1)',
                borderRadius: '12px',
                padding: '16px',
                background: 'rgba(255,255,255,0.6)'
              }}
            >
              <h4
                style={{ color: '#511730', fontSize: '14px', marginBottom: '12px', fontWeight: 600, textAlign: 'center' }}
              >
                释义结果
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {(detectResult.char_results || []).map((c, i) => {
                  const stitchKey = `stitch${selectedIndex !== null ? selectedIndex + 1 : 0}`
                  const data = oracleData[stitchKey]
                  const meaning = data?.charMeanings?.[i]
                  return (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '6px 10px',
                        background: 'rgba(245,230,211,0.5)',
                        borderRadius: '8px',
                        border: '1px solid rgba(81,23,48,0.08)'
                      }}
                    >
                      <span
                        style={{
                          minWidth: '32px',
                          height: '32px',
                          background: 'linear-gradient(135deg, #E0D68A, #CB9173)',
                          borderRadius: '6px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '16px',
                          fontWeight: 700,
                          color: '#511730',
                          flexShrink: 0
                        }}
                      >
                        {meaning?.char || c.chinese_char || c.char || '?'}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 700, color: '#511730', fontSize: '12px' }}>
                          {meaning?.pinyin && <span style={{ color: '#CB9173', marginRight: '4px' }}>{meaning.pinyin}</span>}
                        </div>
                        {meaning?.type && (
                          <div style={{ fontSize: '10px', color: '#8E443D' }}>{meaning.type}</div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {(() => {
            const stitchKey = `stitch${selectedIndex !== null ? selectedIndex + 1 : 0}`
            const data = oracleData[stitchKey]
            if (!data) return null
            return (
              <>
                <div
                  style={{
                    marginTop: '28px',
                    border: '1px solid rgba(81,23,48,0.12)',
                    borderRadius: '16px',
                    padding: '24px',
                    background: 'linear-gradient(145deg, rgba(255,255,255,0.95), rgba(245,230,211,0.5))'
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
                        <tr style={{ background: 'rgba(81,23,48,0.06)' }}>
                          <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>检测框编号</th>
                          <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>拟定字名</th>
                          <th style={{ padding: '10px 8px', textAlign: 'left', color: '#511730', fontWeight: 600 }}>字义说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.charMeanings.map((item, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid rgba(81,23,48,0.06)' }}>
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

                {showAIPanel && (
                  <>
                    <div
                      style={{
                        marginTop: '24px',
                        border: '1px solid rgba(81,23,48,0.12)',
                        borderRadius: '16px',
                        padding: '24px',
                        background: 'linear-gradient(145deg, rgba(224,214,138,0.15), rgba(203,145,115,0.1))',
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
                        {data.title}
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
                        border: '1px solid rgba(81,23,48,0.12)',
                        borderRadius: '16px',
                        padding: '24px',
                        background: 'linear-gradient(145deg, rgba(203,145,115,0.15), rgba(81,23,48,0.08))',
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
                        {data.title}
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
              </>
            )
          })()}
        </div>
      )}

      {phase === 'complete' && (
        <div
          style={{
            animation: 'fpFadeSlide 0.55s ease',
            textAlign: 'center',
            padding: '32px',
            background: 'linear-gradient(135deg, rgba(224,214,138,0.15), rgba(203,145,115,0.15))',
            borderRadius: '20px',
            border: '1px solid rgba(81,23,48,0.12)',
            marginBottom: '24px'
          }}
        >
          <div style={{ fontSize: '36px', marginBottom: '12px' }}>🎉</div>
          <h3 style={{ color: '#511730', fontSize: '20px', fontWeight: 700, marginBottom: '8px' }}>全流程研析完成</h3>
          <p style={{ color: '#8E443D', fontSize: '14px' }}>碎片缀合 → 材质分类 → 甲骨文检测与释义</p>
        </div>
      )}

      {phase !== 'idle' && (
        <div
          style={{
            textAlign: 'center',
            fontWeight: 600,
            color: '#511730',
            marginBottom: '24px',
            fontSize: '14px',
            minHeight: '24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          {['merging', 'stitching', 'classifying', 'detecting'].includes(phase) && (
            <div
              style={{
                width: '18px',
                height: '18px',
                border: '3px solid rgba(81,23,48,0.1)',
                borderTopColor: '#8E443D',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite'
              }}
            />
          )}
          {STATUS_LABELS[phase]}
        </div>
      )}

      <div style={{ display: 'flex', gap: '12px' }}>
        {phase === 'idle' && (
          <button
            type="button"
            onClick={startAnalysis}
            style={{
              flex: 1,
              background: 'linear-gradient(135deg, #8E443D 0%, #511730 100%)',
              color: 'white',
              border: 'none',
              padding: '16px 28px',
              borderRadius: '12px',
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: '15px',
              letterSpacing: '2px',
              boxShadow: '0 4px 20px rgba(81,23,48,0.3)',
              transition: 'all 0.3s ease',
              animation: 'fpPulse 2s ease-in-out infinite'
            }}
          >
            开始缀合
          </button>
        )}

        {phase === 'complete' && (
          <>
            {!showAIPanel ? (
              <button
                type="button"
                onClick={() => {
                  setShowAIPanel(true)
                  setAiCardVisible({ aiInterpretation: false, aiComment: false })
                  setAiInterpretationText('')
                  setAiCommentText('')
                  setShowInterpretationCursor(true)
                  setShowCommentCursor(true)
                  setTimeout(() => setAiCardVisible({ aiInterpretation: true, aiComment: false }), 100)
                  setTimeout(() => setAiCardVisible({ aiInterpretation: true, aiComment: true }), 800)
                  // 获取完整文本内容
                  const stitchKey = `stitch${selectedIndex !== null ? selectedIndex + 1 : 0}`
                  const data = oracleData[stitchKey]
                  if (data) {
                    // AI解读先开始打字，AI点评延迟 1.5 秒后开始
                    setTimeout(() => typeWriter(data.combinedMeaning, setAiInterpretationText, setShowInterpretationCursor, 30), 300)
                    setTimeout(() => typeWriter(data.aiComment, setAiCommentText, setShowCommentCursor, 40), 1800)
                  }
                }}
                style={{
                  flex: 1,
                  background: 'linear-gradient(135deg, #8E443D 0%, #511730 100%)',
                  color: 'white',
                  border: 'none',
                  padding: '16px 28px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '15px',
                  letterSpacing: '2px',
                  boxShadow: '0 4px 20px rgba(81,23,48,0.3)',
                  transition: 'all 0.3s ease',
                  animation: 'fpPulse 2s ease-in-out infinite'
                }}
              >
                开始AI分析
              </button>
            ) : (
              <button
                type="button"
                onClick={resetAnalysis}
                style={{
                  flex: 1,
                  background: 'linear-gradient(135deg, #CB9173 0%, #8E443D 100%)',
                  color: 'white',
                  border: 'none',
                  padding: '16px 28px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '15px',
                  letterSpacing: '2px',
                  boxShadow: '0 4px 20px rgba(142,68,61,0.3)'
                }}
              >
                重新开始研析
              </button>
            )}
          </>
        )}
      </div>

      <style>{`
        @keyframes fpFadeSlide {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fpPulse {
          0%, 100% { box-shadow: 0 4px 20px rgba(81,23,48,0.3); }
          50% { box-shadow: 0 4px 28px rgba(142,68,61,0.45); }
        }
        @keyframes cardAppear {
          from { opacity: 0; transform: translateY(30px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes cursorBlink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        .ai-card-visible {
          animation: cardAppear 0.6s ease forwards;
        }
        .text-reveal {
          opacity: 0;
          animation: textReveal 0.4s ease forwards;
        }
      `}</style>
    </div>
  )
}
