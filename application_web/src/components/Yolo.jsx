import { useState, useRef } from 'react'

export default function Yolo({ baseUrl, db, apiBase }) {
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')
  const [hoveredBox, setHoveredBox] = useState(null)
  const imgRef = useRef(null)

  const moveCarousel = (dir) => {
    setIndex((index + dir + db.yolo.length) % db.yolo.length)
  }

  const runExample = async () => {
    setLoading(true)
    setStatus('正在通过 YOLOv8 检测甲骨文+ViT识别...')
    setResult(null)
    setHoveredBox(null)

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
        return
      }
    }
    setHoveredBox(null)
  }

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
                  onMouseLeave={() => setHoveredBox(null)}
                />
              </div>
            </div>

            {/* 右侧：悬停时显示对应 char_images 图片 */}
            <div style={{
              flex: '0 0 200px',
              minWidth: '200px',
              border: '1px solid rgba(81, 23, 48, 0.1)',
              borderRadius: '16px',
              background: 'linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.5))',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '280px'
            }}>
              <h4 style={{ color: '#511730', fontSize: '15px', marginBottom: '16px', fontWeight: 600 }}>
                对应字图
              </h4>
              {hoveredBox?.char_image ? (
                <img
                  src={'data:image/png;base64,' + hoveredBox.char_image}
                  alt=""
                  style={{
                    maxWidth: '160px',
                    maxHeight: '220px',
                    objectFit: 'contain',
                    borderRadius: '8px',
                    boxShadow: '0 2px 12px rgba(81, 23, 48, 0.15)'
                  }}
                />
              ) : (
                <div style={{
                  color: 'rgba(81, 23, 48, 0.4)',
                  fontSize: '14px',
                  textAlign: 'center',
                  padding: '24px'
                }}>
                  将光标移至左侧检测框上查看对应字图
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
