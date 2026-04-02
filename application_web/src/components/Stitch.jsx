import { useState, useRef } from 'react'

export default function Stitch({ baseUrl, db, apiBase }) {
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('')
  const [previewA, setPreviewA] = useState(null)
  const [previewB, setPreviewB] = useState(null)
  const fileARef = useRef(null)
  const fileBRef = useRef(null)

  const moveCarousel = (dir) => {
    setIndex((index + dir + db.stitch.length) % db.stitch.length)
  }

  const runExample = async () => {
    setLoading(true)
    setStatus('正在下载碎片图并进行几何+网络联合配准，请稍候...')
    setResult(null)
    
    const formData = new FormData()
    formData.append('url_a', baseUrl + db.stitch[index].a)
    formData.append('url_b', baseUrl + db.stitch[index].b)
    
    try {
      const res = await fetch(`${apiBase}/stitch`, {
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

  const handleFileA = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setPreviewA(e.target.result)
      reader.readAsDataURL(file)
    }
  }

  const handleFileB = (e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setPreviewB(e.target.result)
      reader.readAsDataURL(file)
    }
  }

  const handleStitch = async () => {
    const fileA = fileARef.current?.files[0]
    const fileB = fileBRef.current?.files[0]
    if (!fileA || !fileB) {
      setStatus('请先上传本地的碎片 A 和碎片 B！')
      return
    }
    
    setLoading(true)
    setStatus('正在提取边缘并进行联合配准运算，请稍候...')
    setResult(null)
    
    const formData = new FormData()
    formData.append('file_a', fileA)
    formData.append('file_b', fileB)
    
    try {
      const res = await fetch(`${apiBase}/stitch`, {
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
          {db.stitch.map((pair, i) => (
            <div key={i} style={{ minWidth: '100%', boxSizing: 'border-box', padding: '32px', textAlign: 'center' }}>
              <p style={{ fontWeight: 600, marginBottom: '16px', color: '#511730' }}>示例碎片对 {i + 1}</p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '30px' }}>
                <div>
                  <img 
                    src={baseUrl + pair.a} 
                    alt={pair.a}
                    style={{ maxHeight: '180px', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 2px 8px rgba(81, 23, 48, 0.08)', background: 'white', padding: '8px' }}
                  />
                  <br/><small>{pair.a}</small>
                </div>
                <div>
                  <img 
                    src={baseUrl + pair.b} 
                    alt={pair.b}
                    style={{ maxHeight: '180px', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 2px 8px rgba(81, 23, 48, 0.08)', background: 'white', padding: '8px' }}
                  />
                  <br/><small>{pair.b}</small>
                </div>
              </div>
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
        缀合上方展示的在线示例碎片对
      </button>

      <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
        <div 
          onClick={() => fileARef.current?.click()}
          style={{
            flex: 1,
            border: '2px dashed rgba(81, 23, 48, 0.25)',
            padding: '32px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            borderRadius: '16px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(245,230,211,0.4))'
          }}
        >
          <span style={{ fontSize: '36px', marginBottom: '8px', display: 'block' }}>📄</span>
          上传本地碎片 A
          <input 
            type="file" 
            ref={fileARef}
            hidden 
            accept="image/*" 
            onChange={handleFileA} 
          />
          {previewA && (
            <img 
              src={previewA} 
              alt="preview A" 
              style={{ marginTop: '12px', maxHeight: '140px', marginLeft: 'auto', marginRight: 'auto', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} 
            />
          )}
        </div>

        <div 
          onClick={() => fileBRef.current?.click()}
          style={{
            flex: 1,
            border: '2px dashed rgba(81, 23, 48, 0.25)',
            padding: '32px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            borderRadius: '16px',
            background: 'linear-gradient(145deg, rgba(255,255,255,0.6), rgba(245,230,211,0.4))'
          }}
        >
          <span style={{ fontSize: '36px', marginBottom: '8px', display: 'block' }}>📄</span>
          上传本地碎片 B
          <input 
            type="file" 
            ref={fileBRef}
            hidden 
            accept="image/*" 
            onChange={handleFileB} 
          />
          {previewB && (
            <img 
              src={previewB} 
              alt="preview B" 
              style={{ marginTop: '12px', maxHeight: '140px', marginLeft: 'auto', marginRight: 'auto', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }} 
            />
          )}
        </div>
      </div>

      <button 
        onClick={handleStitch}
        disabled={loading}
        style={{
          background: 'linear-gradient(135deg, #CB9173 0%, #8E443D 100%)',
          color: 'white',
          border: 'none',
          padding: '16px 32px',
          borderRadius: '8px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '15px',
          fontWeight: 600,
          width: '100%',
          boxShadow: '0 4px 16px rgba(142, 68, 61, 0.3)',
          opacity: loading ? 0.6 : 1
        }}
      >
        开始智能缀合 (本地图片)
      </button>

      <div style={{ 
        marginTop: '20px',
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
        <div style={{
          marginTop: '24px',
          border: '1px solid rgba(81, 23, 48, 0.1)',
          padding: '24px',
          textAlign: 'center',
          borderRadius: '16px',
          background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5))'
        }}>
          <h4 style={{ color: '#511730', fontSize: '16px', marginBottom: '16px', fontWeight: 600 }}>最终缀合结果</h4>
          <img src={'data:image/jpeg;base64,' + result.result} alt="stitched" style={{ maxWidth: '100%', maxHeight: '450px', borderRadius: '8px' }} />
          <p style={{
            background: 'linear-gradient(135deg, #E0D68A 0%, #CB9173 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '20px',
            fontWeight: 700,
            marginTop: '16px'
          }}>
            最终匹配质量得分: {result.score.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  )
}
