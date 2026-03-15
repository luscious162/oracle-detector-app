import { useState } from 'react'

export default function Classify({ baseUrl, db, apiBase }) {
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')

  const moveCarousel = (dir) => {
    setIndex((index + dir + db.cls.length) % db.cls.length)
  }

  const runExample = async () => {
    setLoading(true)
    setStatus('正在进行分类识别与热图生成...')
    setResult(null)
    
    const formData = new FormData()
    formData.append('url', baseUrl + db.cls[index])
    
    try {
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setStatus('处理完成')
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
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const res = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setStatus('处理完成')
    } catch (e) {
      setStatus('错误: ' + e.message)
    } finally {
      setLoading(false)
    }
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
          {db.cls.map((img, i) => (
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
        识别上方展示的在线示例图片
      </button>

      <div 
        onClick={() => document.getElementById('file-cls').click()}
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
        或点击此处上传本地拓片图片
        <input type="file" id="file-cls" hidden accept="image/*" onChange={handleFile} />
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
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', marginTop: '28px' }}>
          <div style={{
            flex: 1,
            minWidth: '280px',
            border: '1px solid rgba(81, 23, 48, 0.1)',
            padding: '24px',
            textAlign: 'center',
            borderRadius: '16px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5))'
          }}>
            <h4 style={{ color: '#511730', fontSize: '16px', marginBottom: '16px', fontWeight: 600 }}>原始图像</h4>
            <img src={'data:image/png;base64,' + result.original} alt="original" style={{ maxWidth: '100%', maxHeight: '450px', borderRadius: '8px' }} />
          </div>
          <div style={{
            flex: 1,
            minWidth: '280px',
            border: '1px solid rgba(81, 23, 48, 0.1)',
            padding: '24px',
            textAlign: 'center',
            borderRadius: '16px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5))'
          }}>
            <h4 style={{ color: '#511730', fontSize: '16px', marginBottom: '16px', fontWeight: 600 }}>Grad-CAM 注意力热图</h4>
            <img src={'data:image/png;base64,' + result.heatmap} alt="heatmap" style={{ maxWidth: '100%', maxHeight: '450px', borderRadius: '8px' }} />
            <p style={{ fontWeight: 'bold', marginTop: '20px', fontSize: '18px' }}>
              预测: <span style={{
                display: 'inline-block',
                padding: '6px 16px',
                borderRadius: '20px',
                fontWeight: 600,
                background: result.label === 'Bone' ? 'linear-gradient(135deg, #E3F2FD, #BBDEFB)' : 'linear-gradient(135deg, #FCE4EC, #F8BBD9)',
                color: result.label === 'Bone' ? '#1565C0' : '#C2185B'
              }}>{result.label}</span>
              <span style={{ fontSize: '14px', color: '#888', marginTop: '8px', display: 'block' }}>
                Bone: {(result.prob_bone * 100).toFixed(1)}% | Shell: {(result.prob_shell * 100).toFixed(1)}%
              </span>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
