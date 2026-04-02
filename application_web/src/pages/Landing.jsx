import { useNavigate } from 'react-router-dom'
import { MeshGradient } from '@paper-design/shaders-react'
import './Landing.css'

export default function Landing() {
  const navigate = useNavigate()

  const features = [
    { icon: '📊', label: '分类识别' },
    { icon: '🔗', label: '碎片缀合' },
    { icon: '🖋️', label: '摹本生成' },
    { icon: '🔎', label: '甲骨文检测与释义' }
  ]

  return (
    <div
      className="landing-bg"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 20px'
      }}
    >
      <MeshGradient
        speed={1}
        scale={1}
        distortion={0.8}
        swirl={0.1}
        colors={['#E0D68A', '#CB9173', '#8E443D', '#511730']}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 0
        }}
      />

      <div
        style={{
          position: 'relative',
          zIndex: 1,
          textAlign: 'center',
          padding: '40px',
          maxWidth: '800px',
          width: '100%'
        }}
      >
        <div
          className="landing-card"
          style={{
            background: 'rgba(255, 255, 255, 0.92)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            padding: '60px 80px',
            borderRadius: '24px',
            boxShadow: '0 16px 48px rgba(81, 23, 48, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.5)'
          }}
        >
          <h1
            className="landing-title"
            style={{
              fontSize: '42px',
              fontWeight: 700,
              background: 'linear-gradient(135deg, #511730 0%, #8E443D 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '6px',
              marginBottom: '16px'
            }}
          >
            甲骨识微
          </h1>

          <p
            className="landing-subtitle"
            style={{
              color: '#5A5A5A',
              fontSize: '16px',
              letterSpacing: '3px',
              marginBottom: '48px'
            }}
          >
            ORACLE BONE INSCRIPTION INTELLIGENT ANALYSIS
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '16px',
              marginBottom: '32px'
            }}
          >
            {features.map((item) => (
              <div
                key={item.label}
                className="landing-feature-item"
                style={{
                  padding: '16px',
                  background: 'linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5))',
                  borderRadius: '12px',
                  border: '1px solid rgba(81, 23, 48, 0.1)',
                  cursor: 'default'
                }}
              >
                <span style={{ fontSize: '28px', display: 'block', marginBottom: '4px' }}>
                  {item.icon}
                </span>
                <p style={{ margin: 0, fontWeight: 600, color: '#511730' }}>{item.label}</p>
              </div>
            ))}
          </div>

          <button
            type="button"
            className="landing-cta"
            onClick={() => navigate('/main')}
            style={{
              background: 'linear-gradient(135deg, #8E443D 0%, #511730 100%)',
              color: 'white',
              border: 'none',
              padding: '18px 48px',
              borderRadius: '12px',
              fontSize: '18px',
              fontWeight: 600,
              cursor: 'pointer',
              letterSpacing: '2px',
              boxShadow: '0 8px 24px rgba(81, 23, 48, 0.35)'
            }}
          >
            开始使用
          </button>
        </div>
      </div>
    </div>
  )
}
